from ts2d import TS2D


def main():
    model = TS2D()
    res = model.predict(r"C:\datasets\mimas\tots_001_ct.nii.gz", ofile=r"C:\test\test.nrrd")
    print("Test successful!")

if __name__ == '__main__':
    main()