import spacy
from spacy.util import minibatch, compounding
import random
import json
import os
from pathlib import Path
from spacy.training import offsets_to_biluo_tags
def validate_entities(nlp, text, entities):
    """Validate entity alignment with the text."""
    doc = nlp.make_doc(text)
    valid = True
    non_overlapping_entities = []
    try:
        biluo_tags = offsets_to_biluo_tags(doc, entities)
    except ValueError as e:
        print(f"Entities: {entities}")
        print(f"Error: {e}")
        valid = False

    # Check if entity offsets are within bounds
    for start, end, label in entities:
        if start < 0 or end > len(text):
            print(f"Entity out of bounds: {start}-{end}, {label}, in text: {text}")
            valid = False

    # Filter out overlapping entities
    for new_start, new_end, label in entities:
        if new_start >= 0 and new_end <= len(text) and all(new_start >= end or new_end <= start for start, end, _ in non_overlapping_entities):
            non_overlapping_entities.append((new_start, new_end, label))
        else:
            print(f"Skipping invalid entity: {new_start}-{new_end}, {label}, in text: {text[new_start:new_end]}")

    return valid and len(non_overlapping_entities) > 0
def prepare_training_data(json_data):
    """Convert JSON data to spaCy training format."""
    text = json_data.get("content", "").strip()
    if not text:
        print("Skipping example with empty text.")
        return None

    annotations = {"entities": []}
    annotation_list = json_data.get("annotation", [])
    if not isinstance(annotation_list, list):
        print(f"Invalid or missing 'annotation' key in JSON data: {json_data}")
        return None

    for annotation in annotation_list:
        for point in annotation.get("points", []):
            start = point["start"]
            end = point["end"]
            label = annotation.get("label", [])
            if not label or not isinstance(label, list):
                print(f"Skipping invalid label in annotation: {annotation}")
                continue
            label = label[0]  # Assuming a single label per annotation
            if start < 0 or end > len(text):
                print(f"Skipping invalid entity: {start}-{end}, {label}, in text: {text[start:end]}")
                continue
            annotations["entities"].append((start, end, label))

    if not annotations["entities"]:
        print(f"No valid entities found in text: {text}")
        return None

    return (text, annotations)
def load_json_file(file_path):
    """Load a JSON file or line-delimited JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)  # Reset file pointer
            if first_char == '[':  # Standard JSON array
                return json.load(f)
            else:  # Line-delimited JSON
                return [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def train_ner_model(training_data, output_dir, n_iter=30):
    """Train a custom NER model."""
    # Create output directory
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # Load model or create blank
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Loaded model 'en_core_web_sm'")
    except OSError:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")
    
    # Create or get NER component
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
    
    # Add entity labels
    entity_types = set()
    for _, annotations in training_data:
        for _, _, label in annotations.get("entities", []):
            entity_types.add(label)
    
    for entity_type in entity_types:
        ner.add_label(entity_type)
    
    print(f"Training with {len(training_data)} examples and {len(entity_types)} entity types")
    print(f"Entity types: {', '.join(entity_types)}")
    
    # Get names of other pipes to disable during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    # Only train NER
    with nlp.disable_pipes(*other_pipes), nlp.select_pipes(enable=["ner"]):
        # Reset and initialize weights
        if "ner" in nlp.pipe_names:
            nlp.begin_training()
        
        # Training loop
        for itn in range(n_iter):
            random.shuffle(training_data)
            losses = {}
            
            for text, annotations in training_data:
                doc = nlp.make_doc(text)
                try:
                    biluo_tags = offsets_to_biluo_tags(doc, annotations["entities"])
                    # Validate entity offsets before generating BILUO tags
                    if all(0 <= start < end <= len(text) for start, end, _ in annotations["entities"]):
                        biluo_tags = offsets_to_biluo_tags(doc, annotations["entities"])
                    else:
                        print(f"Skipping invalid entity offsets in text: {text}")
                        continue
                except ValueError as e:
                    print(f"Error generating BILUO tags for text: {text}. Error: {e}")
                    continue

            # Batch up the examples
            batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    examples.append(Example.from_dict(doc, annotations))

                # Update model
                nlp.update(
                    examples,  # Pass a batch of Example objects
                    drop=0.5,  # Dropout to prevent overfitting
                    losses=losses
                )
            
            if itn % 5 == 0:
                print(f"Iteration {itn}, Losses: {losses}")
    
    # Save model
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")
    
    return nlp


def main():
    # Parameters
    json_file_path = "data/json/traindata.json"  # Path to the single JSON file
    output_dir = "models/resume_ner_model"
    n_iter = 30

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the single JSON file
    json_data = load_json_file(json_file_path)
    if not json_data:
        print(f"Failed to load JSON file: {json_file_path}")
        return

    # Prepare training data
    training_data = []
    for item in json_data:  # Iterate over the list of JSON objects
        example = prepare_training_data(item)
        if example:
            text, annotations = example
            try:
                nlp = spacy.blank("en")  # Use a blank model for validation
                if validate_entities(nlp, text, annotations["entities"]):
                    training_data.append(example)
                else:
                    print("Skipping misaligned example.")
            except Exception as e:
                print(f"Error during validation: {e}")
        else:
            print("No valid training data found in the JSON file.")

    print(f"Prepared {len(training_data)} training examples")

    if not training_data:
        print("No valid training data found.")
        return

    # Train model
    model = train_ner_model(training_data, output_dir, n_iter=n_iter)

    # Test on a sample
    sample = """
    Afreen Jamadar
    Active member of IIIT Committee in Third year
    Sangli, Maharashtra - Email me on Indeed: indeed.com/r/Afreen-Jamadar/8baf379b705e37c6
    """
    doc = model(sample)
    print("\nEntities in sample text:")
    for ent in doc.ents:
        print(f"{ent.text} - {ent.label_} ({ent.start_char}:{ent.end_char})")

if __name__ == "__main__":
    main()