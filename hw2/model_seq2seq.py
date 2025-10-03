import argparse, os, json, math
import numpy as np
import torch
import torch.nn as nn
import h5py

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Vocab:
    def __init__(self, itos, stoi):
        self.itos = itos
        self.stoi = stoi
    def decode(self, ids):
        toks = []
        for i in ids:
            if isinstance(i, torch.Tensor):
                i = i.item()
            if 0 <= i < len(self.itos):
                tok = self.itos[i]
                if tok not in ["<PAD>", "<BOS>", "<EOS>"]:
                    toks.append(tok)
        return " ".join(toks)

class Encoder(nn.Module):
    def __init__(self, feat_dim, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(feat_dim, hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers>1 else 0)
    def forward(self, x):
        outputs, h = self.rnn(x)
        return outputs, h

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.scale = 1.0 / math.sqrt(hidden_size)
    def forward(self, dec_h, enc_out):
        scores = torch.bmm(enc_out, dec_h.unsqueeze(2)).squeeze(2) * self.scale
        weights = torch.softmax(scores, dim=1)
        ctx = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)
        return ctx, weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, emb_size=256, num_layers=1, dropout=0.1, use_attention=True, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_id)
        self.rnn = nn.GRU(emb_size + (hidden_size if use_attention else 0), hidden_size,
                           num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers>1 else 0)
        self.use_attention = use_attention
        self.attn = Attention(hidden_size) if use_attention else None
        self.out = nn.Linear(hidden_size, vocab_size)

    def step(self, enc_outputs, h, token):
        emb = self.emb(token)
        if self.use_attention:
            ctx, _ = self.attn(h[-1], enc_outputs)
            rnn_in = torch.cat([emb, ctx], dim=-1).unsqueeze(1)
        else:
            rnn_in = emb.unsqueeze(1)
        out, h2 = self.rnn(rnn_in, h)
        logits = self.out(out.squeeze(1))
        return logits, h2

class S2VT(nn.Module):
    def __init__(self, feat_dim, vocab_size, hidden=256, emb=256, enc_layers=1, dec_layers=1,
                 dropout=0.1, use_attention=True, pad_id=0):
        super().__init__()
        self.encoder = Encoder(feat_dim, hidden, enc_layers, dropout)
        self.decoder = Decoder(vocab_size, hidden, emb, dec_layers, dropout, use_attention, pad_id=pad_id)

    @torch.no_grad()
    def greedy_decode(self, x, max_len, bos_id, eos_id):
        enc_out, h = self.encoder(x)
        B, T, H = enc_out.size()
        token = torch.full((B,), bos_id, dtype=torch.long, device=x.device)
        outputs = []
        for _ in range(max_len):
            logits, h = self.decoder.step(enc_out, h, token)
            token = torch.argmax(logits, dim=-1)
            outputs.append(token)
            if (token == eos_id).all():
                break
        outs = torch.stack(outputs, dim=1) if outputs else torch.zeros((B,0), dtype=torch.long, device=x.device)
        seq = outs[0].tolist() if outs.size(0) > 0 else []
        if eos_id in seq:
            seq = seq[:seq.index(eos_id)]
        return seq

def load_meta(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta["itos"], meta["stoi"], meta["cfg"], meta["feat_dim"]

def build_model(itos, stoi, cfg, feat_dim):
    pad_id = stoi.get("<PAD>", 0)
    model = S2VT(
        feat_dim=feat_dim,
        vocab_size=len(itos),
        hidden=cfg["hidden"],
        emb=cfg["emb"],
        enc_layers=cfg["enc_layers"],
        dec_layers=cfg["dec_layers"],
        dropout=cfg["dropout"],
        use_attention=cfg["use_attention"],
        pad_id=pad_id,
    ).to(DEVICE)
    return model

def load_weights(model, h5_path):
    state = {}
    with h5py.File(h5_path, "r") as hf:
        for k in hf.keys():
            state[k] = torch.tensor(hf[k][:])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)
    model.eval()
    return model

def list_video_ids(feat_dir):
    vids = []
    for fname in os.listdir(feat_dir):
        if fname.endswith(".npy"):
            vids.append(os.path.splitext(fname)[0])
    vids.sort()
    return vids

def load_feat(path):
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 3:
        T = arr.shape[0]
        arr = arr.reshape(T, -1)
    return torch.tensor(arr, dtype=torch.float32)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--max_len", type=int, default=25)
    args = ap.parse_args()

    feat_dir = os.path.join(args.data_dir, "feat")
    if not os.path.isdir(feat_dir):
        raise FileNotFoundError(f"Feature directory not found: {feat_dir}")

    itos, stoi, cfg, feat_dim = load_meta(args.meta)
    model = build_model(itos, stoi, cfg, feat_dim)
    model = load_weights(model, args.weights)

    bos_id = stoi.get("<BOS>", 1)
    eos_id = stoi.get("<EOS>", 2)

    # simple vocab obj for decoding
    class _V: pass
    vocab = _V()
    vocab.itos = itos

    vids = list_video_ids(feat_dir)
    results = []
    for vid in vids:
        x = load_feat(os.path.join(feat_dir, vid + ".npy")).unsqueeze(0).to(DEVICE)
        seq = model.greedy_decode(x, max_len=args.max_len, bos_id=bos_id, eos_id=eos_id)
        # decode via itos, skipping specials
        toks = [itos[i] for i in seq if 0 <= i < len(itos) and itos[i] not in ["<PAD>", "<BOS>", "<EOS>"]]
        caption = " ".join(toks)
        results.append((vid, caption))

    with open(args.output, "w") as f:
        for vid, cap in results:
            f.write(f"{vid},{cap}\n")
    print(f"[OK] wrote predictions to {args.output} ({len(results)} lines)")

if __name__ == "__main__":
    main()
