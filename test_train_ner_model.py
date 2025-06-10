#!/usr/bin/env python
# coding: utf8
from __future__ import unicode_literals
from __future__ import print_function
import re
import plac
import random
from pathlib import Path
import spacy
from spacy.training.example import Example
import json
import logging

# new entity label
LABEL = "COL_NAME"

from spacy.training.iob_utils import offsets_to_biluo_tags

def validate_training_data(nlp, training_data):
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        try:
            tags = offsets_to_biluo_tags(doc, annotations["entities"])
        except Exception as e:
            print(f"Error in text: {text}")
            print(f"Entities: {annotations['entities']}")
            print(e)

def remove_overlapping_entities(training_data):
    """Remove overlapping entities from the training data."""
    cleaned_data = []
    for text, annotations in training_data:
        entities = annotations["entities"]
        non_overlapping_entities = []
        entities = sorted(entities, key=lambda x: x[0])  # Sort by start position
        prev_end = -1
        for start, end, label in entities:
            if start >= prev_end:  # No overlap
                non_overlapping_entities.append((start, end, label))
                prev_end = end
            else:
                print(f"Removed overlapping entity: {(start, end, label)} in text: {text}")
        cleaned_data.append((text, {"entities": non_overlapping_entities}))
    return cleaned_data

def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans."""
    invalid_span_tokens = re.compile(r'\s')  # Matches whitespace characters

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end

            # Adjust the start position to skip leading whitespace
            while valid_start < len(text) and invalid_span_tokens.match(text[valid_start]):
                valid_start += 1

            # Adjust the end position to skip trailing whitespace
            while valid_end > valid_start and invalid_span_tokens.match(text[valid_end - 1]):
                valid_end -= 1

            # Only add the entity if the span is valid
            if valid_start < valid_end:
                valid_entities.append((valid_start, valid_end, label))
            else:
                print(f"Removed invalid entity span: ({start}, {end}, {label}) in text: {text}")

        cleaned_data.append((text, {'entities': valid_entities}))

    return cleaned_data


def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
    try:
        training_data = []
        with open(dataturks_JSON_FilePath, 'r', encoding="utf8") as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            if data['annotation'] is not None:
                for annotation in data['annotation']:
                    point = annotation['points'][0]
                    labels = annotation['label']
                    if not isinstance(labels, list):
                        labels = [labels]

                    for label in labels:
                        entities.append((
                            point['start'],
                            point['end'] + 1,
                            label
                        ))

            training_data.append((text, {"entities": entities}))
        return training_data
    except Exception:
        logging.exception("Unable to process " + dataturks_JSON_FilePath)
        return None


TRAIN_DATA = trim_entity_spans(convert_dataturks_to_spacy("data/json/traindata.json"))
TRAIN_DATA = remove_overlapping_entities(TRAIN_DATA)
# Validate the training data
validate_training_data(spacy.blank("en"), TRAIN_DATA)

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(
    model=None,
    new_model_name="training",
    output_dir='models/resume_ner_model/',
    n_iter=1
):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")



    # Add entity recognizer to model if it's not in the pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Add labels to the NER pipeline
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # Disable other pipes during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            print(f"Starting iteration {itn}")
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update(
                    [example],
                    drop=0.2,
                    losses=losses,
                )
            print("Losses", losses)

    # Test the trained model
    test_text = "Marathwada Mitra Mandals College of Engineering"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # Save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # Test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print("Label",ent.label_,"Text", ent.text)


if __name__ == "__main__":
    plac.call(main)
