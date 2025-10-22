Data Passing Guide
==================

Understanding Data Flow in Collie
----------------------------------

Collie uses an event-driven architecture where components communicate through **Event** objects. 
Each Event contains a **Payload** with typed fields and an optional **Context** for metadata.

User-Friendly Features
-----------------------

Collie Payloads are designed with developer experience in mind:

✅ **Type Safety**: Standard fields have clear types for IDE autocomplete

✅ **Flexibility**: ``extra_data`` field for custom data without breaking changes

✅ **Helper Methods**: Convenient methods for accessing extra data:
   
   - ``payload.get_extra("key", default)`` - Safe access with default
   - ``payload.set_extra("key", value)`` - Fluent setter with chaining
   - ``payload.has_extra("key")`` - Check existence

✅ **Method Chaining**: Build payloads fluently:

.. code-block:: python

   payload = (TransformerPayload(train_data=df)
              .set_extra("feature_names", features)
              .set_extra("n_classes", 3))

✅ **Pydantic Validation**: Automatic validation and serialization

Three Ways to Pass Data
------------------------

1. Standard Payload Fields (Recommended for Common Data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the predefined fields in each Payload type for standard ML pipeline data:

**TransformerPayload**

.. code-block:: python

   TransformerPayload(
       train_data=pd.DataFrame,      # Training dataset
       validation_data=pd.DataFrame,  # Validation dataset
       test_data=pd.DataFrame         # Test dataset
   )

**TunerPayload**

.. code-block:: python

   TunerPayload(
       hyperparameters=dict,          # Best hyperparameters found
       train_data=pd.DataFrame,       # Pass along training data
       validation_data=pd.DataFrame,  # Pass along validation data
       test_data=pd.DataFrame         # Pass along test data
   )

**TrainerPayload**

.. code-block:: python

   TrainerPayload(
       model=Any  # Trained model object
   )
   
   # Optional: Use extra_data for framework-specific metrics
   TrainerPayload(
       model=model,
       extra_data={
           "train_loss": 0.05,  # PyTorch/TensorFlow loss
           "val_loss": 0.08,
           "epochs": 100
       }
   )

**EvaluatorPayload**

.. code-block:: python

   EvaluatorPayload(
       metrics=List[Dict],            # List of metric dictionaries
       is_better_than_production=bool # Whether to promote model
   )

**PusherPayload**

.. code-block:: python

   PusherPayload(
       model_uri=str,                 # MLflow model URI
       status=str,                    # Deployment status
       model_version=str              # Model version number
   )

2. Extra Data Field (For Custom/Experimental Data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every Payload has an ``extra_data`` dictionary field for flexible custom data.

**Three Ways to Use Extra Data:**

.. code-block:: python

   # Method 1: Direct dictionary access
   payload = TransformerPayload(
       train_data=df,
       extra_data={"feature_names": ["age", "income"]}
   )
   
   # Method 2: Using helper methods (recommended for better readability)
   payload = TransformerPayload(train_data=df)
   payload.set_extra("feature_names", ["age", "income"])
   payload.set_extra("n_classes", 3)
   
   # Method 3: Method chaining
   payload = (TransformerPayload(train_data=df)
              .set_extra("feature_names", ["age", "income"])
              .set_extra("n_classes", 3)
              .set_extra("data_source", "database"))

**Helper Methods:**

.. code-block:: python

   # Set a value
   payload.set_extra("key", value)  # Returns self for chaining
   
   # Get a value with default
   value = payload.get_extra("key", default_value)
   
   # Check if key exists
   if payload.has_extra("key"):
       value = payload.get_extra("key")

**Example: Passing Feature Engineering Metadata**

.. code-block:: python

   class MyTransformer(Transformer):
       def handle(self, event: Event) -> Event:
           # Process data
           train_data, feature_info = preprocess_data()
           
           return Event(
               payload=TransformerPayload(
                   train_data=train_data,
                   validation_data=None,
                   test_data=None,
                   extra_data={
                       "feature_names": ["age", "income", "score"],
                       "categorical_features": ["gender", "city"],
                       "numeric_features": ["age", "income"],
                       "encoding_mappings": {"city": {"NY": 0, "LA": 1}},
                       "scaler_params": {"mean": 0.5, "std": 0.2}
                   }
               )
           )

**Example: Accessing and Extending Extra Data**

.. code-block:: python

   class MyTrainer(Trainer):
       def handle(self, event: Event) -> Event:
           # Get data from previous component
           train_data = event.payload.train_data
           
           # Access extra data using helper methods (recommended)
           feature_names = event.payload.get_extra("feature_names", [])
           categorical_features = event.payload.get_extra("categorical_features", [])
           
           # Or check if exists first
           if event.payload.has_extra("scaler_params"):
               scaler_params = event.payload.get_extra("scaler_params")
           
           # Use the information
           self.mlflow.log_params({
               "n_features": len(feature_names),
               "n_categorical": len(categorical_features)
           })
           
           # Train model
           model = train_model(train_data, feature_names)
           
           # Build new payload with extra data - three approaches:
           
           # Approach 1: Create with dict (merge previous extra_data)
           return Event(
               payload=TrainerPayload(
                   model=model,
                   extra_data={
                       **event.payload.extra_data,  # Keep previous
                       "training_time_seconds": 120.5,
                       "n_epochs": 50,
                       "train_loss": 0.1,  # Optional metrics
                       "val_loss": 0.15
                   }
               )
           )
           
           # Approach 2: Use helper methods (more readable)
           payload = TrainerPayload(
               model=model,
               extra_data=event.payload.extra_data.copy()  # Copy previous
           )
           payload.set_extra("training_time_seconds", 120.5)
           payload.set_extra("n_epochs", 50)
           payload.set_extra("train_loss", 0.1)
           payload.set_extra("val_loss", 0.15)
           return Event(payload=payload)
           
           # Approach 3: Method chaining (most concise)
           payload = (TrainerPayload(model=model, 
                                    extra_data=event.payload.extra_data.copy())
                      .set_extra("training_time_seconds", 120.5)
                      .set_extra("n_epochs", 50)
                      .set_extra("train_loss", 0.1)
                      .set_extra("val_loss", 0.15)
                      .set_extra("early_stopping_epoch", 35))
           return Event(payload=payload)

**Example: Evaluation with Custom Metrics**

.. code-block:: python

   class MyEvaluator(Evaluator):
       def handle(self, event: Event) -> Event:
           model = event.payload.model
           test_data = event.payload.test_data
           
           # Perform evaluation
           metrics = evaluate_model(model, test_data)
           
           # Save detailed reports
           report_path = "evaluation_report.html"
           generate_report(metrics, report_path)
           self.mlflow.log_artifact(report_path)
           
           return Event(
               payload=EvaluatorPayload(
                   metrics=[metrics],
                   is_better_than_production=metrics["accuracy"] > 0.9,
                   extra_data={
                       "report_path": report_path,
                       "confusion_matrix": metrics["confusion_matrix"].tolist(),
                       "per_class_metrics": metrics["per_class"],
                       "roc_auc_scores": metrics["roc_auc"],
                       "evaluation_time": "2024-01-15T10:30:00"
                   }
               )
           )

3. Event Context (Framework Internal - Rarely Needed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   **Most users should NOT use event.context!**
   
   Context is primarily for framework internal use. Use ``payload.extra_data`` for custom data instead.

**When Context is Appropriate (Rare Cases):**

Only use ``event.context`` for pipeline execution metadata that framework or monitoring tools need:

.. code-block:: python

   class MyComponent(Transformer):
       def handle(self, event: Event) -> Event:
           import time
           start_time = time.time()
           
           # Do work
           result = process_data()
           
           # ⚠️ Use context ONLY for execution metadata
           event.context.set("processing_time", time.time() - start_time)
           event.context.set("component_version", "1.2.0")
           event.context.set("hostname", socket.gethostname())
           event.context.set("execution_timestamp", datetime.now().isoformat())
           
           # ✅ All business data goes in payload
           return Event(
               payload=TransformerPayload(
                   train_data=result,
                   extra_data={
                       "feature_names": features,  # ← Use extra_data, not context!
                       "n_samples": len(result)
                   }
               ),
               context=event.context
           )

**❌ DON'T Use Context For:**

- Model objects
- Training data
- Feature names
- Hyperparameters
- Metrics
- Any business logic data

**✅ DO Use Context For:**

- Execution timestamps
- Processing duration
- Component versions
- Debug flags
- Pipeline metadata

**Better Alternative - Use extra_data:**

.. code-block:: python

   # ❌ Wrong - putting business data in context
   event.context.set("feature_names", features)
   
   # ✅ Correct - use payload.extra_data
   payload.set_extra("feature_names", features)

Decision Tree: Which Method to Use?
------------------------------------

.. code-block:: text

   Is it core ML data (model, datasets, metrics)?
   ├─ YES → Use standard Payload fields
   │         Example: train_data, model, metrics
   │
   └─ NO → Is it custom pipeline data?
           ├─ YES → Use payload.extra_data (RECOMMENDED)
           │         Example: feature_names, training_curves, custom_metrics
           │
           └─ NO → Is it framework/execution metadata?
                   └─ YES → Use event.context (RARELY NEEDED)
                             Example: processing_time, execution_timestamp
                             Note: Most users should use extra_data instead!

Common Patterns
---------------

Pattern 1: Passing Data Through the Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Transformer creates data
   TransformerPayload(
       train_data=train_df,
       extra_data={"feature_names": features}
   )
   
   # Tuner passes it along with hyperparameters
   TunerPayload(
       hyperparameters=best_params,
       train_data=event.payload.train_data,  # Pass through
       extra_data=event.payload.extra_data   # Pass through
   )
   
   # Trainer uses it
   train_data = event.payload.train_data
   features = event.payload.extra_data["feature_names"]

Pattern 2: Accumulating Extra Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Each component adds to extra_data
   return Event(
       payload=SomePayload(
           ...,
           extra_data={
               **event.payload.extra_data,  # Keep previous data
               "my_new_field": my_value      # Add new data
           }
       )
   )

Pattern 3: Conditional Data Passing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MyTrainer(Trainer):
       def handle(self, event: Event) -> Event:
           extra_data = {}
           
           # Only include if debugging
           if self.debug_mode:
               extra_data["training_curve"] = training_history
               extra_data["gradient_norms"] = gradient_norms
           
           return Event(
               payload=TrainerPayload(
                   model=model,
                   extra_data=extra_data
               )
           )

Best Practices
--------------

DO's ✅
~~~~~~~

- **Use standard fields** for common ML data (datasets, models, metrics)
- **Use extra_data** for custom, experimental, or pipeline-specific data
- **Use event.context** only for processing metadata (timestamps, versions)
- **Document** what you put in extra_data in your component docstrings
- **Provide defaults** when accessing extra_data: ``extra_data.get("key", default)``
- **Keep extra_data serializable** (use built-in types, not complex objects)

DON'Ts ❌
~~~~~~~~~

- **Don't** put large objects in extra_data (use artifacts instead)
- **Don't** rely on undocumented extra_data fields from other components
- **Don't** use context for main business data (use payload instead)
- **Don't** modify event.payload.extra_data in place (create a new dict)
- **Don't** put model objects in extra_data (use standard model field)

Complete Example
----------------

Here's a complete pipeline showing all three data passing methods:

.. code-block:: python

   from collie import Event
   from collie.core import (
       Transformer, Trainer, Evaluator,
       TransformerPayload, TrainerPayload, EvaluatorPayload
   )
   import time

   class DataTransformer(Transformer):
       def handle(self, event: Event) -> Event:
           start = time.time()
           
           # Load and process data
           train_data, metadata = load_and_process()
           
           # Context: processing metadata
           event.context.set("transform_time", time.time() - start)
           event.context.set("data_version", "v2.0")
           
           return Event(
               payload=TransformerPayload(
                   # Standard fields
                   train_data=train_data,
                   validation_data=None,
                   test_data=None,
                   # Extra data: feature engineering info
                   extra_data={
                       "feature_names": metadata["features"],
                       "encoding_maps": metadata["encodings"],
                       "outliers_removed": metadata["outliers_count"]
                   }
               ),
               context=event.context
           )

   class ModelTrainer(Trainer):
       def handle(self, event: Event) -> Event:
           start = time.time()
           
           # Standard fields
           train_data = event.payload.train_data
           
           # Extra data
           features = event.payload.extra_data.get("feature_names", [])
           
           # Context
           data_version = event.context.get("data_version")
           self.mlflow.log_param("data_version", data_version)
           
           # Train
           model, history = train_model(train_data, features)
           
           # Context: training metadata
           event.context.set("training_time", time.time() - start)
           
           return Event(
               payload=TrainerPayload(
                   # Standard field
                   model=model,
                   # Extra data: optional metrics and training details
                   extra_data={
                       **event.payload.extra_data,  # Keep previous
                       "train_loss": history["loss"][-1],
                       "val_loss": history["val_loss"][-1],
                       "training_curve": history["loss"],
                       "best_epoch": history["best_epoch"],
                       "optimizer": "Adam"
                   }
               ),
               context=event.context
           )

   class ModelEvaluator(Evaluator):
       def handle(self, event: Event) -> Event:
           model = event.payload.model
           
           # Evaluate
           metrics = evaluate(model)
           
           return Event(
               payload=EvaluatorPayload(
                   metrics=[metrics],
                   is_better_than_production=metrics["accuracy"] > 0.9,
                   extra_data={
                       "detailed_report": "report.html",
                       "confusion_matrix": metrics["cm"].tolist()
                   }
               )
           )

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 20 25 35 20

   * - Method
     - Use For
     - Example
     - Usage Frequency
   * - **Standard Fields**
     - Core ML data
     - ``train_data``, ``model``, ``metrics``
     - Always
   * - **extra_data**
     - Custom/experimental data
     - ``feature_names``, ``training_curves``, ``custom_reports``
     - Common
   * - **event.context**
     - Framework metadata only
     - ``processing_time``, ``execution_timestamp``
     - Rarely (internal use)

**Quick Reference:**

- 🟢 **Use Standard Fields**: For all core ML data (model, datasets, metrics)
- 🟡 **Use extra_data**: For custom pipeline data (99% of custom needs)
- 🔴 **Avoid context**: Only for framework/execution metadata (rarely needed)

For more examples, see the :doc:`core_concepts` and :doc:`mlflow_integration` pages.
