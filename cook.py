from subprocess import check_call


def main():
    check_call(
        "sphinx-build source build --fail-on-warning --keep-going --nitpicky",
        shell=True,
    )


if __name__ == "__main__":
    main()
