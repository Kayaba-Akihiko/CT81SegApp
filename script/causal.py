


from xmodules.xutils import lib_utils

def main():
    from lightning import Fabric
    print(Fabric)

    print(lib_utils.import_available('Fabric', 'lightning'))
    print(lib_utils.import_available('lightning.Fabric'))


if __name__ == '__main__':
    main()