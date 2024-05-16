def calculate_bpp(unit, weight, dimensions):
    # Convertir le poids de l'image en bits
    if unit == "Mo":
        weight *= 8 * 1024 * 1024
    elif unit == "ko":
        weight *= 8 * 1024
    elif unit == "octet":
        weight *= 8
    elif unit == "bit":
        pass  # Pas besoin de conversion si le poids est déjà en bits

    dim_x, dim_y = dimensions
    # Calculer le nombre total de pixels
    total_pixels = dim_x * dim_y
    # Calculer le bpp
    bpp = weight / total_pixels

    return bpp

def main():
    unit = input("Entrez l'unité de poids (Mo, ko, octet ou bit): ")
    weight = float(input("Entrez le poids de l'image: "))
    dimensions = tuple(map(int, input("Entrez les dimensions de l'image sous forme de tuple (dim_x, dim_y): ").split(',')))

    bpp = calculate_bpp(unit, weight, dimensions)
    print("Le nombre de bits par pixel (bpp) est de:", bpp)

if __name__ == "__main__":
    main()
