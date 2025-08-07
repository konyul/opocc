#!/usr/bin/env python3
"""Check the structure of the pickle annotation file."""

import pickle
import sys

if len(sys.argv) != 2:
    print("Usage: python check_data.py <pickle_file_path>")
    sys.exit(1)

pickle_file = sys.argv[1]

try:
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded pickle file: {pickle_file}")
    print(f"Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        
        if 'data_list' in data:
            data_list = data['data_list']
            print(f"\ndata_list length: {len(data_list)}")
            
            if len(data_list) > 0:
                print(f"\nFirst sample keys:")
                first_sample = data_list[0]
                for key, value in first_sample.items():
                    print(f"  {key}: {value if len(str(value)) < 100 else str(value)[:100] + '...'}")
        
        if 'metainfo' in data:
            print(f"\nmetainfo: {data['metainfo']}")
    
except Exception as e:
    print(f"Error loading pickle file: {e}")
    import traceback
    traceback.print_exc()