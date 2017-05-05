from classifier import train_evaluate
import numpy as np
import PIL


def eval(pred_op, image_path):
    label_lookup = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    image_mean   = 133.0
    image_data   = np.array(PIL.Image.open(image_path), dtype=np.float32)
    image_data  -= image_mean
    image_data   = np.ascontiguousarray(np.transpose(image_data, (2, 0, 1)))
    
    result       = np.squeeze(pred_op.eval({pred_op.arguments[0]:[image_data]}))
    
    # Return top 3 results:
    top_count = 3
    result_indices = (-np.array(result)).argsort()[:top_count]

    print("Top 3 predictions:")
    for i in range(top_count):
        print("\tLabel: {:10s}, confidence: {:.2f}%".format(label_lookup[result_indices[i]], result[result_indices[i]] * 100))


def main():       
    c_obj = train_evaluate(5,50000,16)
    c_obj.train()
    eval(c_obj.call_me(),"data/CIFAR-10/test/00014.png")

if __name__=="__main__":
    main()
