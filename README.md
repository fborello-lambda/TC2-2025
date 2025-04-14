# TC2-2025

Create a virtual environment and install the `requirements.txt`.

Use Python's venv module to create a new virtual environment:

```sh
python3 -m venv venv
```

Activate the Virtual Environment:

### On Linux/Mac:
```sh
source venv/bin/activate
```

### On Windows:
```sh

.\venv\Scripts\activate
```

### Install Project Dependencies

Once the virtual environment is activated, install the packages specified in the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

Then, if using VSCode, select the venv as the kernel.
