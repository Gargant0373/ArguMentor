def main():
    from src.dataset import download_dataset
    data = download_dataset()
    print(data.head())

if __name__ == "__main__":
    main()