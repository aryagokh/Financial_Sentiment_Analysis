model_path = './src/finetuned_bert_for_financial_sentiment_analysis'

def map_to_class(pred):
    maps = {1 : 'Positive 😊', 2 : 'Negative 😓', 0 : 'Neutral 😐'}
    return maps.get(pred)