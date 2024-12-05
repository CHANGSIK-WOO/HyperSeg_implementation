import os
import json
import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--upload-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    df = pd.read_table(args.annotation_file)

    cur_df = df.copy()
    cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
    cur_df.insert(6, 'prediction', None)
    for pred in open(args.result_dir):
        pred = json.loads(pred)
        option_char = [k for j in pred['option_char'] for k in j]

        # pred_text = pred['text']
        
        if pred['text'] in option_char:
            pred_text = pred['text']
        elif len(pred['text']) > 1 and pred['text'][0] in option_char:
            pred_text = pred['text'][0]

        else:
            pred_text = 'A'

        cur_df.loc[df['index'] == pred['question_id'], 'prediction'] = pred_text

    cur_df.to_excel(os.path.join(args.upload_dir, f"{args.experiment}.xlsx"), index=False, engine='openpyxl')
