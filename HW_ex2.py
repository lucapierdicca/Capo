import numpy as np
from pprint import pprint
#=========== Direct Sampling ===========

#2x2    f t
#    f |_|_|
#	 t | | |

tr_model = np.array([[0.7,0.3],
					 [0.3,0.7]])
em_model = np.array([[0.8,0.2],
					 [0.1,0.9]])
R_0 = np.array([0.5,0.5])


sequences = np.zeros((40*2,20),dtype=int)
for i in range(0,sequences.shape[0],2):
	for j in range(sequences.shape[1]):
		if j==0:
			sequences[i,j] = np.random.binomial(1,R_0[0])
		else:
			sequences[i,j] = np.random.binomial(1,tr_model[sequences[i,j-1],1])

		sequences[i+1,j] = np.random.binomial(1,em_model[sequences[i,j],1])


#============ Forward-Backward =============

em_model_t = np.diag(np.diag(em_model@np.array([[0,0],[1,1]])))
em_model_f = np.diag(np.diag(em_model@np.array([[1,1],[0,0]])))

#FILTERING (and PREDICTION)
def forward(evidence):

	curr_state_est = R_0.reshape((2,1))

	f_h = []

	for i in evidence:
		new_evidence = i
		if new_evidence == 1:
			new_state_est = em_model_t@tr_model.T@curr_state_est
		else:
			new_state_est = em_model_f@tr_model.T@curr_state_est

		new_state_est = new_state_est/np.sum(new_state_est)

		curr_state_est = new_state_est

		f_h.append(curr_state_est)


	return f_h


def backward(evidence):

	curr_back_msg = np.array([[1.0],[1.0]])

	b_h = [curr_back_msg]

	for i in range(len(evidence),1,-1):
		new_evidence = evidence[i-1]
		if new_evidence == 1:
			new_back_msg = tr_model@em_model_t@curr_back_msg
		else:
			new_back_msg = tr_model@em_model_f@curr_back_msg

		curr_back_msg = new_back_msg

		b_h.insert(0,curr_back_msg)
	
	#curr_back_msg = curr_back_msg/np.sum(curr_back_msg)

	return b_h

#SMOOTHING
def forward_backward(evidence):
	f_h = forward(evidence)
	b_h = backward(evidence)

	f_b_h = []
	for f,b in zip(f_h,b_h):
		curr_state_est = f*b
		curr_state_est = curr_state_est/np.sum(curr_state_est)
		f_b_h.append(curr_state_est)

	return f_b_h



#============= Improvement 1 ===============

def forward_backward_imp(evidence):
	f_h = forward(evidence)
	
	curr_for_msg = f_h[-1]
	curr_back_msg = np.array([[1.0],[1.0]])

	f_b_h = [curr_for_msg*curr_back_msg]
	for i in range(len(evidence),1,-1):
		new_evidence = evidence[i-1]

		if new_evidence == 1:
			new_back_msg = tr_model@em_model_t@curr_back_msg
			new_for_msg = np.linalg.inv(tr_model.T)@np.linalg.inv(em_model_t)@curr_for_msg
		else:
			new_back_msg = tr_model@em_model_f@curr_back_msg
			new_for_msg = np.linalg.inv(tr_model.T)@np.linalg.inv(em_model_f)@curr_for_msg


		new_for_msg = new_for_msg/np.sum(new_for_msg)

		curr_for_msg = new_for_msg
		curr_back_msg = new_back_msg

		curr_state_est = curr_for_msg*curr_back_msg
		curr_state_est = curr_state_est/np.sum(curr_state_est)

		f_b_h.insert(0,curr_state_est)

	return f_b_h


#AIMA example check
# ev = np.array([1,1,1,1])
# filtered = forward(ev)
# backward_msg = backward(ev)
# smoothed = forward_backward(ev)
# smoothed_imp = forward_backward_imp(ev)

# pprint(filtered)
# pprint(backward_msg)
# pprint(smoothed)
# pprint(smoothed_imp)



#============= FILTERING vs. SMOOTHING ==============


for i in range(0,sequences.shape[0],2):
	test = sequences[i,:]
	evidence = sequences[i+1,:]

	proba_f = forward(evidence)
	#proba_fb = forward_backward(evidence)
	proba_fb_imp = forward_backward_imp(evidence)
	results = ["%d -- %.2f - %.2f" % (k,i[1,0],j[1,0]) for i,j,k in zip(proba_f,proba_fb_imp,test)]

	print('State sequence: ',test)
	print(results)
	print()



