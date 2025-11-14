#!/bin/bash

# Run the first Python script
python3 fine_tune_vilt.py

# Check if the first script ran successfully
if [ $? -eq 0 ]; then
  echo "fine_tune_vilt.py ran successfully"
else
  echo "fine_tune_vilt.py failed"
  exit 1
fi

# Run the second Python script
python3 fine_tune_lxmert.py

# Check if the second script ran successfully
if [ $? -eq 0 ]; then
  echo "fine_tune_lxmert.py ran successfully"
else
  echo "fine_tune_lxmert.py failed"
  exit 1
fi

echo "Both scripts completed successfully!"
