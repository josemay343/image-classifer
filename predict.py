import time
import argparse
import util_functions
import json
from args_input import args_input


def main():
    
    start_time = time.time()

    in_arg = args_input()

    
    print('Loading trained model...')
    model, optimizer, criterion, device = util_functions.load_checkpoint(in_arg.checkpoint, in_arg.gpu)
   
    print('\nPredicting the input image classification...\n')  
    probs, flower_name = util_functions.predict(in_arg.input, model, device, in_arg.topk, in_arg.category_name)
 
    print('Based on the input image the following classification was determined:\n')
    for (prob, flower) in zip(probs, flower_name):
        print('Flower: {} - Probability: {:.2f}%'.format(flower, prob*100))
    
    print('\nTime taken to predict: {:.1f}s'.format(time.time() - start_time))
          
# Call to main function to run the program
if __name__ == "__main__":
    main()

