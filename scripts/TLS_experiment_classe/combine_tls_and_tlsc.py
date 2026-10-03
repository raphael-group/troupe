import pickle 

path_to_tls =   "/Users/william_hs/Desktop/Projects/troupe/experiments/TLS_no_endothelial/processed_data/trees.pkl"
path_to_tlsc =  "/Users/william_hs/Desktop/Projects/troupe/experiments/TLSC/processed_data/trees.pkl"
out_path =      "/Users/william_hs/Desktop/Projects/troupe/experiments/TLS_and_TLSC/processed_data/trees.pkl"

with open(path_to_tls, "rb") as fp:
    tls_list = pickle.load(fp)
with open(path_to_tlsc, "rb") as fp:
    tlsc_list = pickle.load(fp)

with open(out_path, "wb") as fp:
    pickle.dump(tls_list + tlsc_list, fp)