from transformers import ElectraTokenizer, ElectraForQuestionAnswering, pipeline

def run_pipeline():
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v2-distilled-korquad-384")
    model = ElectraForQuestionAnswering.from_pretrained("monologg/koelectra-small-v2-distilled-korquad-384")

    qa = pipeline("question-answering", tokenizer=tokenizer, model=model, device=0)
    model.cuda()

    os.makedirs('out', exist_ok=True)
    with torch.no_grad(), open('out/baseline.csv', 'w') as fd:
        writer = csv.writer(fd)
        writer.writerow(['Id', 'Predicted'])

        rows = []
        for sample in tqdm(test_dataset, "Testing"):
            answer = qa({
                "question": sample['question'],
                "context": sample['context'],
            })['answer']

            rows.append([sample["guid"], answer])
        
        writer.writerows(rows)