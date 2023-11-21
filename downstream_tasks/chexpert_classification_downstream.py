def get_chexpert_prompts_bin(preds_history, col_names):

    for idx, pred in enumerate(preds_history):
        questions = []
        for disease in col_names:
            #if disease not in ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]:
            # if disease == "No Finding" or disease == "Pleural Other":
            question_prompt = " Is there any " + disease + "?"
            pred = pred.replace("ASSISTANT:", "ASSISTANT: ")
            full_prompt = pred + "</s>USER: " + question_prompt + " ASSISTANT:"

            questions.append(full_prompt)

        preds_history[idx] = questions

    return preds_history

def get_chexpert_prompts_all(preds_history, col_names):

    for idx, pred in enumerate(preds_history):
        question = "List all the findings in this report."
        pred = pred.replace("ASSISTANT:", "ASSISTANT: ")
        full_prompt = pred + "</s>USER: " + question + " ASSISTANT:"
        preds_history[idx] = full_prompt

    return preds_history


if __name__ == '__main__':
    pass