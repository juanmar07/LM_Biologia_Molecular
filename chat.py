from generar import cargar_modelo, generar_texto

def main():
    modelo, tokenizador = cargar_modelo()
    print("\nMINI GPT EDUCATIVO - escribe 'salir' para terminar\n")
    while True:
        pregunta=input("Usuario: ").strip()
        if pregunta.lower() in {"salir","exit","quit"}:
            break
        prompt=f"Usuario: {pregunta}\nAsistente:"
        salida=generar_texto(modelo,tokenizador,prompt)
        print("\nModelo:")
        print(salida)
        print()

if __name__ == "__main__":
    main()
