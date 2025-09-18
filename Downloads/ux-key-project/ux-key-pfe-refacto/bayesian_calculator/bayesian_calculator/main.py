


import sys
import warnings
from request_processor import *



def main():

    '''
    Main function to execute the processing pipeline.
    '''


    
    warnings.filterwarnings('ignore')

    if len(sys.argv) < 3:
        print("Usage: python3 process.py <data_path> <yaml_path>")
        sys.exit(1)

    data_path = sys.argv[1]
    yaml_config_filename = sys.argv[2]


    processor = RequestProcessor(yaml_config_filename)
    
   
    out = processor.process_all()

    for i, process_out in enumerate(out) :
        print(f"--------------------- Process n° : {i+1} started --------------------- ")
        output_processor, model, score, metric, bounds = process_out
        
        data = read_data(data_path,metric, bounds)
       
        model.update_params(data)
        if output_processor.type == 'CI':
            output = output_processor.process(model, score, data)
        elif output_processor.type == 'PDF': 
            output = output_processor.process(model)
        else :
            output = output_processor.process(model, data)
        
        print(f"--------------------- Process n° : {i+1} ENDED --------------------- ")
    return 
if __name__ == '__main__':
    main()

