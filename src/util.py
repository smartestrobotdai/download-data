import hashlib
import os.path

def md5(in_str):
	m = hashlib.sha256()
	b = bytearray(in_str, 'utf-8')
	m.update(b)
	return m.hexdigest()


def remove_centralized(data):
	assert(data.shape[1]==516)
	return data[:,7:-5]

def timestamp2date(timestamp):
	return timestamp.date().strftime("%y%m%d")

def get_latest_dir(save_path):
    all_subdirs = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))]
    max_time = 0
    for dirname in all_subdirs:
        fullname = os.path.join(save_path, dirname)
        time = os.path.getmtime(fullname)
        if time > max_time:
            max_time = time
            result = dirname
    return result

def get_earliest_dir(save_path):
    all_subdirs = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))]
    max_time = 0
    for dirname in all_subdirs:
        fullname = os.path.join(save_path, dirname)
        time = os.path.getmtime(fullname)
        if time > max_time:
            max_time = time
            result = dirname
    return result