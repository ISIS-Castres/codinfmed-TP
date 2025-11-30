import subprocess
from transformers import pipeline
from gliner import GLiNER
import json


class NERModel:
    def __init__(self, name, backend="gliner", uniner_model_path=None):
        """
        name: model identifier (HF name or GLiNER checkpoint)
        backend: "gliner", "nunER", or "uniner"
        uniner_model_path: path to the UniNER model if using backend="uniner"
        """
        self.name = name
        self.backend = backend
        self.uniner_model_path = uniner_model_path

        if backend == "gliner":
            self.model = GLiNER.from_pretrained(name)
        elif backend == "nunER":
            self.model = pipeline(
                "token-classification", model=name, aggregation_strategy="simple"
            )
        elif backend == "uniner":
            if uniner_model_path is None:
                raise ValueError("For UniNER, you must provide uniner_model_path")
            # nothing to load in Python; CLI will be called at predict time
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def predict_entities(self, text, labels=None, threshold=0.5):
        if self.backend == "gliner":
            ents = self.model.predict_entities(text, labels, threshold=threshold)
            return ents

        elif self.backend == "nunER":
            preds = self.model(text)
            return [
                {
                    "label": ent["entity_group"],
                    "start": ent["start"],
                    "end": ent["end"],
                    "score": ent["score"],
                }
                for ent in preds
            ]

        elif self.backend == "uniner":
            # Call UniNER CLI via subprocess
            cmd = ["python", "src/serve/cli.py", "--model_path", self.uniner_model_path]
            proc = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            output = proc.stdout.decode("utf-8")

            try:
                preds = json.loads(output)
            except json.JSONDecodeError:
                print("UniNER output could not be parsed as JSON:")
                return []

            # Convert UniNER output to unified format
            entities = []
            for ent in preds:
                entities.append(
                    {
                        "label": ent.get("label", "UNK"),
                        "start": ent.get("start", 0),
                        "end": ent.get("end", 0),
                        "score": ent.get("score", 1.0),
                    }
                )
            return entities
