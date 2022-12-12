import os
import pathlib
import pickle


def create_dir(list_path):
    for path in list_path:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def save_fit_history(fold, fit, path):
    filename = os.path.join(path, f'fold{fold}-fit.pckl')
    try:
        with open(filename, 'wb') as file:
            pickle.dump(fit.history, file)
            file.close()
            print(f'{filename} created')
    except Exception:
        raise SystemExit(f'error in create {filename}')


