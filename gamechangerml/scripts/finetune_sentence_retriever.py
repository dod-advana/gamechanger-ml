from gamechangerml.src.search.sent_transformer.finetune import STFinetuner
from gamechangerml.configs.config import ValidationConfig, EmbedderConfig
from gamechangerml.api.utils.pathselect import get_model_paths
from gamechangerml.api.utils.logger import logger

model_path_dict = get_model_paths()

ES_URL = 'https://vpc-gamechanger-iquxkyq2dobz4antllp35g2vby.us-east-1.es.amazonaws.com'
VALIDATION_DIR = ValidationConfig.DATA_ARGS['validation_dir']
LOCAL_TRANSFORMERS_DIR = model_path_dict["transformers"]
BASE_MODEL_NAME = EmbedderConfig.MODEL_ARGS['model_name']


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Finetuning the sentence transformer model with in-domain data")
    
    parser.add_argument(
        "--data-path", "-d", 
        dest="data_path", 
        help="path to csv with finetuning data"
        )

    parser.add_argument(
        "--model-path", "-m", 
        dest="model_path", 
        help="path to model for fine-tuning"
        )

    args = parser.parse_args()

    
    logger.info("|------------------Collecting training data------------------|")

    if args.data_path:
        training_data_csv_path = args.data_path
        df = pd.read_csv(training_data_csv_path)
    else:
        training_data_csv_path = os.path.join(VALIDATION_DIR, timestamp_filename('finetune_sent_data', '.csv'))
        df = collect_training_data() 

    logger.info("|---------------------Splitting train/test-------------------|")
    split_ratio = EmbedderConfig.MODEL_ARGS['train_proportion']
    train, test = split_train_test(df, split_ratio)

    logger.info("|---------------------Finetuning model-----------------------|")
    if args.model_path:
        model_load_path = args.model_path
    else:
        model_load_path = os.path.join(LOCAL_TRANSFORMERS_DIR, BASE_MODEL_NAME)

    model_save_path = os.path.join(LOCAL_TRANSFORMERS_DIR, timestamp_filename(BASE_MODEL_NAME + '_finetuned', '/'))

    ## load original model
    model = SentenceTransformer(model_load_path)
    train, test = finetune(train, test, model=model, model_save_path=model_save_path, **EmbedderConfig.MODEL_ARGS['finetune'])

    all_data = pd.concat([train, test])
    all_data.to_csv(training_data_csv_path)
    logger.info("Training data saved to {}".format(str(training_data_csv_path)))