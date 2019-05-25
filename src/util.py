import hashlib

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