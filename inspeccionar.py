from tokenizador import TokenizadorEducativo
from dataset import GestorDataset

def main():
    tokenizador=TokenizadorEducativo(); tokenizador.cargar()
    frase="el servidor apache funciona correctamente"
    print("TEXTO:",frase)
    print("TOKENS:",tokenizador.separar(frase))
    ids=tokenizador.codificar(frase)
    print("IDS:",ids)
    print("DECODE:",tokenizador.decodificar(ids))
    dataset=GestorDataset(); x,y=dataset.obtener_batch("train")
    print("SHAPE X:",tuple(x.shape))
    print("PRIMER X:",x[0].tolist())
    print("PRIMER Y:",y[0].tolist())

if __name__ == "__main__":
    main()
