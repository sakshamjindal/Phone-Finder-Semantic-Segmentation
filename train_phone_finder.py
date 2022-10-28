
import sys
from phone_finder import PhoneFinder

params = {
    "num_epochs" : 15,
    "num_classes": 2,
    "batch_size": 4,
    "num_workers": 2,
    "log_frequency" : 1,
    "device" : "cuda",
    "ce_weights" : [1, 10],
    "image_size" : (256, 256)
}

if __name__ == '__main__':
    try:
        folder = sys.argv[1]
    except Exception as e:
        print(sys.argv)
        print(e)

    phone_finder = PhoneFinder()
    phone_finder.train_phone_finder(params)


