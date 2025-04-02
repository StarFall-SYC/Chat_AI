# AI Chat Model - Project Structure

## Optimized Project Structure

The project has been restructured to follow a more standardized organization and naming convention. All filenames are now in English for better compatibility across different platforms and development environments.

## Directory Structure

```
ai_chat_model/
├── data/                  # Data related resources
│   ├── training/          # Training data
│   │   ├── intents.json   # Basic intent data
│   │   └── extended_intents.json  # Extended intent data
│   ├── models/            # Model storage directory
│   ├── history/           # Conversation history
│   ├── __init__.py        # Package initialization
│   └── data_loader.py     # Data loading utilities
│
├── models/                # Model implementation
│   ├── __init__.py        # Package initialization
│   ├── chatbot.py         # Chatbot model implementation
│   └── text_processor.py  # Text processing utilities
│
├── ui/                    # User interface components
│   ├── __init__.py        # Package initialization
│   ├── main_window.py     # Main window implementation
│   ├── chat_tab.py        # Chat interface tab
│   └── training_tab.py    # Training interface tab
│
├── scripts/               # Utility scripts
│   ├── run_demo.py        # Full application launcher
│   ├── start_chat.py      # Chat interface launcher
│   ├── start_training.py  # Training interface launcher
│   ├── test_chatbot.py    # Command-line chat testing
│   ├── test_trainer.py    # Training test script
│   └── quick_train.py     # Quick model training
│
├── docs/                  # Project documentation
│   ├── README.md          # Project overview
│   ├── FEATURES.md        # Detailed feature description
│   ├── QUICKSTART.md      # Quick start guide
│   ├── SUMMARY.md         # Project summary
│   ├── TECHNICAL.md       # Technical documentation (was "项目文档.md")
│   └── PROJECT_STRUCTURE.md  # This file - structure documentation
│
├── __init__.py            # Package initialization
├── main.py                # Application entry point
├── requirements.txt       # Project dependencies
├── LICENSE                # MIT License
└── .gitignore             # Git ignore rules
```

## Naming Conventions

The project follows these naming conventions:

1. **Files and Directories**: All files and directories use lowercase with underscores (_) for spaces.
   - Exception: Documentation files use uppercase with no spaces (e.g., README.md)

2. **Python Modules**: All Python modules use lowercase with underscores.
   - Example: `data_loader.py`, `text_processor.py`

3. **Python Classes**: Classes use CamelCase (PascalCase).
   - Example: `ChatModel`, `TextProcessor`

4. **Python Functions**: Functions use lowercase with underscores.
   - Example: `load_training_data()`, `get_response()`

5. **Documentation**: Documentation files use uppercase with .md extension.
   - Example: `README.md`, `FEATURES.md`

## Migration Guide

When the project was restructured, the following changes were made:

1. Created a `scripts/` directory for utility scripts
2. Created a `docs/` directory for documentation
3. Renamed "项目文档.md" to "TECHNICAL.md"
4. Added this "PROJECT_STRUCTURE.md" file

## Benefits of the New Structure

1. **Improved Navigation**: Clear separation between code, scripts, and documentation
2. **Cross-platform Compatibility**: English filenames work better across different OSes
3. **Standardization**: Following common project structure patterns 
4. **Maintainability**: Easier for new developers to understand the codebase
5. **Documentation Organization**: All documentation now resides in a dedicated directory 