from .service import run


def main() -> None:
    result = run("  alice  ")
    print(result)


if __name__ == "__main__":
    main()

