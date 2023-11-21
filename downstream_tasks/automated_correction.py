from chexbert.run_chexbert import run_chexbert_labeler

def get_correction_prompts(preds_history, col_names, chexpert_preds, chexpert_labels):

    false_positives = chexpert_preds * (1 - chexpert_labels)
    false_negatives = (1 - chexpert_preds) * chexpert_labels

    for idx, (fp, fn) in enumerate(zip(false_positives, false_negatives)):
        fp = [col_names[i] for i, v in enumerate(fp) if v == 1]
        fn = [col_names[i] for i, v in enumerate(fn) if v == 1]
        if "No Finding" in fp:
            fp.remove("No Finding")
        if "No Finding" in fn:
            fn.remove("No Finding")
        fp_str = ', '.join(fp)
        fp_str = fp_str.rsplit(', ', 1)
        fp_str = ' and '.join(fp_str)
        fn_str = ', '.join(fn)
        fn_str = fn_str.rsplit(', ', 1)
        fn_str = ' and '.join(fn_str)

        if len(fp) > 0 and len(fn) > 0:
            corr_prompt = f"Please adapt the report with the following corrections: Include {fn_str.lower()} and remove {fp_str.lower()}. Don't make other changes."
        elif len(fp) > 0:
            corr_prompt = f"The patient does not have {fp_str.lower()}. Update the report. Don't make other changes."
        elif len(fn) > 0:
            corr_prompt = f"The patient also has {fn_str.lower()}, correct the report. Don't make other changes."
        else:
            corr_prompt = "KEEP_OLD"

        # add space after ASSISTANT:
        preds_history[idx] = preds_history[idx].replace("ASSISTANT:", "ASSISTANT: ")
        preds_history[idx] += "</s>USER: " + corr_prompt + " ASSISTANT:"

    return preds_history

def get_correction_labels(col_names, chexpert_preds, chexpert_labels):

    false_positives = chexpert_preds * (1 - chexpert_labels)
    false_negatives = (1 - chexpert_preds) * chexpert_labels

    all_fps = []
    all_fns = []
    for idx, (fp, fn) in enumerate(zip(false_positives, false_negatives)):
        fp = [col_names[i] for i, v in enumerate(fp) if v == 1]
        fn = [col_names[i] for i, v in enumerate(fn) if v == 1]
        if "No Finding" in fp:
            fp.remove("No Finding")
        if "No Finding" in fn:
            fn.remove("No Finding")
        all_fps.append(fp)
        all_fns.append(fn)

    return all_fps, all_fns


if __name__ == '__main__':
    pass