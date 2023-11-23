from Visualize_Output import *
from Training import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    choice = input("Want to train new model or check model output?\nTo Train new model write 'train', To check model output write 'test':")
    if choice == 'train':
        train_model()
    elif choice == 'test':
        Visualize_Depth()
    else:
        print("Wrong input")
