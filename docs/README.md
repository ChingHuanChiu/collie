# Sphinx API Documentation Guide

## 📚 Overview

This directory contains the Sphinx documentation for Collie, including comprehensive API documentation with MLflow integration examples.

## 🚀 Quick Start

### Build Documentation

```bash
# Navigate to docs directory
cd docs

# Build HTML documentation
make html

# View the documentation
open _build/html/index.html  # macOS
# or
xdg-open _build/html/index.html  # Linux
```

### Clean Build

```bash
make clean
make html
```

## 📖 Documentation Structure

```
docs/
├── conf.py                     # Sphinx configuration
├── index.rst                   # Documentation home page
├── getting_started.rst         # Getting started guide
├── core_concepts.rst           # Core concepts and architecture
├── mlflow_integration.rst      # MLflow integration guide (detailed!)
├── api/
│   ├── core.rst               # Core API reference
│   └── contracts.rst          # Contracts API reference
├── _build/                    # Generated HTML (git ignored)
└── Makefile                   # Build commands
```

## 🎯 Key Features

### 1. Automatic API Documentation

Sphinx automatically generates API docs from docstrings:

```python
class Transformer(MLflowIntegration):
    """
    Base class for data transformation.
    
    Examples:
        >>> class MyTransformer(Transformer):
        ...     def transform(self, context):
        ...         self.mlflow.log_param("n_samples", len(data))
        ...         return {"data": transformed_data}
    """
```

### 2. MLflow Methods Documentation

Complete documentation of all `self.mlflow` methods available in components:

- `mlflow_integration.rst` - Comprehensive guide with examples
- Shows all logging methods (params, metrics, artifacts)
- Shows model management (log, load, register)
- Shows data logging
- Complete working examples

### 3. Type Hints Support

The `sphinx-autodoc-typehints` extension automatically documents type hints:

```python
def log_param(self, key: str, value: Union[str, int, float]) -> None:
    """Log a parameter."""
```

### 4. Code Examples

All documentation includes working code examples showing:
- How to use each component
- How to use MLflow methods
- Complete pipeline examples

## 📝 What's Documented

### Core Components (api/core.rst)

- **Transformer** - Data preprocessing with MLflow logging
- **Trainer** - Model training with hyperparameter logging
- **Tuner** - Hyperparameter optimization with nested runs
- **Evaluator** - Model evaluation with metrics logging
- **Pusher** - Model deployment with registry integration
- **Orchestrator** - Pipeline orchestration

Each component includes:
- Class documentation
- Available MLflow methods
- Usage examples
- Common patterns

### MLflow Integration (mlflow_integration.rst)

Comprehensive guide covering:

1. **Core Logging Methods**
   - Parameters (`log_param`, `log_params`)
   - Metrics (`log_metric`, `log_metrics`)
   - Artifacts (`log_artifact`, `log_artifacts`, `log_dict`)

2. **Model Management**
   - Log models (auto-detection of sklearn, pytorch, etc.)
   - Load models from registry
   - Register models

3. **Data Logging**
   - Log input datasets for lineage

4. **Advanced Features**
   - Nested runs for experiments
   - Model comparison
   - Tagging and organization

5. **Complete Examples**
   - Full pipeline with all MLflow methods
   - Real-world usage patterns

### Contracts (api/contracts.rst)

- **MLflowIntegration** - Base class providing MLflow access
- **Event System** - Event types and handling
- **PipelineContext** - Context object for data flow

Includes complete reference table of all MLflow methods.

## 🔧 Customization

### Adding New Pages

1. Create a new `.rst` file
2. Add it to `index.rst` in the appropriate `toctree`
3. Rebuild: `make html`

Example:

```rst
.. toctree::
   :maxdepth: 2
   
   getting_started
   core_concepts
   your_new_page  # Add here
```

### Documenting New Components

Add autodoc directive to appropriate `.rst` file:

```rst
.. autoclass:: collie.your_module.YourClass
   :members:
   :show-inheritance:
```

### Updating MLflow Methods

Edit `mlflow_integration.rst` or `api/contracts.rst` to add new examples or methods.

## 📊 Build Output

After running `make html`, you'll find:

```
_build/html/
├── index.html              # Home page
├── getting_started.html    # Getting started guide
├── core_concepts.html      # Architecture guide
├── mlflow_integration.html # MLflow guide (important!)
├── api/
│   ├── core.html          # Core API
│   └── contracts.html     # Contracts API
├── genindex.html          # General index
├── search.html            # Search page
└── _static/               # CSS, JS, images
```

## 🌐 Publishing Documentation

### Option 1: Read the Docs

1. Connect your GitHub repo to [Read the Docs](https://readthedocs.org)
2. RTD will automatically build from `docs/`
3. Documentation will be available at: `https://<your-username>-collie.readthedocs.io`

### Option 2: GitHub Pages

```bash
# Install ghp-import
pip install ghp-import

# Build and push to gh-pages branch
make html
ghp-import -n -p -f _build/html
```

Documentation available at: `https://yourusername.github.io/collie`

### Option 3: Host Locally

```bash
# Serve documentation locally
cd _build/html
python -m http.server 8000

# Visit: http://localhost:8000
```

## 📖 Documentation Best Practices

### 1. Write Good Docstrings

```python
class MyComponent(Transformer):
    """
    Short description.
    
    Longer description with more details about what this does
    and how to use it.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Dictionary containing processed data
        
    Examples:
        >>> component = MyComponent()
        >>> result = component.transform(context)
        >>> result["data"]
        
    Note:
        Important information about usage.
        
    Warning:
        Things to watch out for.
    """
```

### 2. Include Examples

Every component and method should have:
- Basic usage example
- MLflow logging example
- Complete working example

### 3. Document MLflow Usage

For each component, document:
- Which MLflow methods are typically used
- What should be logged (params, metrics, artifacts)
- Example code showing the logging

### 4. Keep It Updated

When adding new features:
1. Update docstrings
2. Add examples to relevant `.rst` files
3. Rebuild docs: `make html`

## 🐛 Troubleshooting

### Import Errors

If you see import errors when building:

```bash
# Make sure collie is importable
cd ..
python -c "import collie; print('OK')"

# Or install in development mode
pip install -e .
```

### Missing Modules

```bash
# Install missing Sphinx extensions
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
```

### Warnings

Sphinx will show warnings for:
- Missing docstrings
- Broken references
- Invalid RST syntax

Fix these to improve documentation quality.

## 📚 Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Guide](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Read the Docs](https://docs.readthedocs.io/)
- [Sphinx AutoDoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)

## 🎉 What You Get

With this Sphinx setup, you get:

✅ **Complete API Reference** - All classes, methods, and functions documented
✅ **MLflow Methods** - Comprehensive guide to all `self.mlflow` methods
✅ **Usage Examples** - Working code examples throughout
✅ **Search Functionality** - Full-text search of documentation
✅ **Cross-References** - Links between related documentation
✅ **Professional Look** - Clean Read the Docs theme
✅ **Easy to Maintain** - Auto-generated from docstrings
✅ **Versioned Docs** - Support for multiple versions (RTD)

## 🚀 Next Steps

1. **Review the Generated Docs**
   ```bash
   cd docs
   make html
   open _build/html/index.html
   ```

2. **Check the MLflow Integration Guide**
   - Open `_build/html/mlflow_integration.html`
   - This has all the `self.mlflow` methods documented!

3. **Customize**
   - Update `conf.py` with your project info
   - Add more examples to `.rst` files
   - Improve docstrings in code

4. **Publish**
   - Set up Read the Docs or GitHub Pages
   - Share the docs link in your README

---

**Happy Documenting! 📚✨**
