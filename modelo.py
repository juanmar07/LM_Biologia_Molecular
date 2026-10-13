import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LONGITUD_CONTEXTO, DIMENSION_EMBEDDING, NUM_CABEZAS, NUM_CAPAS, DIMENSION_FEED_FORWARD, DROPOUT

class AtencionCausal(nn.Module):
    def __init__(self):
        super().__init__()
        if DIMENSION_EMBEDDING % NUM_CABEZAS != 0:
            raise ValueError("DIMENSION_EMBEDDING debe ser divisible por NUM_CABEZAS")
        self.atencion = nn.MultiheadAttention(DIMENSION_EMBEDDING, NUM_CABEZAS, dropout=DROPOUT, batch_first=True)
        # True bloquea posiciones futuras; el buffer sigue al modelo al cambiar de dispositivo.
        self.register_buffer(
            "mascara_causal",
            torch.triu(torch.ones(LONGITUD_CONTEXTO, LONGITUD_CONTEXTO, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        longitud = x.size(1)
        mascara = self.mascara_causal[:longitud, :longitud]
        salida, _ = self.atencion(x, x, x, attn_mask=mascara, need_weights=False)
        return salida

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(DIMENSION_EMBEDDING, DIMENSION_FEED_FORWARD),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(DIMENSION_FEED_FORWARD, DIMENSION_EMBEDDING),
            nn.Dropout(DROPOUT),
        )
    def forward(self, x):
        return self.red(x)

class BloqueTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIMENSION_EMBEDDING)
        self.norm2 = nn.LayerNorm(DIMENSION_EMBEDDING)
        self.atencion = AtencionCausal()
        self.feed_forward = FeedForward()
    def forward(self, x):
        x = x + self.atencion(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, tamanio_vocabulario):
        super().__init__()
        self.tamanio_vocabulario = tamanio_vocabulario
        self.embedding_tokens = nn.Embedding(tamanio_vocabulario, DIMENSION_EMBEDDING)
        self.embedding_posiciones = nn.Embedding(LONGITUD_CONTEXTO, DIMENSION_EMBEDDING)
        self.dropout = nn.Dropout(DROPOUT)
        self.bloques = nn.ModuleList([BloqueTransformer() for _ in range(NUM_CAPAS)])
        self.norm_final = nn.LayerNorm(DIMENSION_EMBEDDING)
        self.salida_vocabulario = nn.Linear(DIMENSION_EMBEDDING, tamanio_vocabulario, bias=False)
        self.salida_vocabulario.weight = self.embedding_tokens.weight
        self._inicializar_pesos()

    def _inicializar_pesos(self):
        """Política 3A.1 B: escala pequeña sin reinicializar el peso compartido."""
        pesos_inicializados = set()
        for modulo in self.modules():
            if isinstance(modulo, (nn.Linear, nn.Embedding)):
                identidad = id(modulo.weight)
                if identidad not in pesos_inicializados:
                    nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
                    pesos_inicializados.add(identidad)
                if isinstance(modulo, nn.Linear) and modulo.bias is not None:
                    nn.init.zeros_(modulo.bias)
            elif isinstance(modulo, nn.LayerNorm):
                nn.init.ones_(modulo.weight)
                nn.init.zeros_(modulo.bias)
        # MultiheadAttention.in_proj_weight no es nn.Linear: conserva Xavier.

    def forward(self, indices, objetivos=None):
        _, longitud = indices.shape
        if longitud > LONGITUD_CONTEXTO:
            raise ValueError(f"La secuencia supera LONGITUD_CONTEXTO={LONGITUD_CONTEXTO}")
        posiciones = torch.arange(longitud, device=indices.device)
        x = self.dropout(self.embedding_tokens(indices) + self.embedding_posiciones(posiciones))
        for bloque in self.bloques:
            x = bloque(x)
        x = self.norm_final(x)
        logits = self.salida_vocabulario(x)
        perdida = None
        if objetivos is not None:
            perdida = F.cross_entropy(logits.reshape(-1, self.tamanio_vocabulario), objetivos.reshape(-1))
        return logits, perdida
