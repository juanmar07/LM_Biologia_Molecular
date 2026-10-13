from preparar_corpus import preparar_corpus
from tokenizador import entrenar_tokenizador_y_tokenizar
from entrenar import entrenar

def main():
    print("PASO 1/3 - Preparar corpus")
    preparar_corpus()
    print("\nPASO 2/3 - Entrenar tokenizador")
    entrenar_tokenizador_y_tokenizar()
    print("\nPASO 3/3 - Entrenar Transformer")
    entrenar()
    print("\nPipeline finalizado. Ejecuta python generar.py o python chat.py")

if __name__ == "__main__":
    main()
