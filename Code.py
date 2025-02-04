import numpy as np

########################## DATA EXTRACTION ###########################################

''' We first begin by extracting all the data in our problem set into the form of matrices A,b,c so that it can be used for data processing easily. The data extraction is done through simple file reading operations with the pre assumed knowledge that total constraints = 500 and total decision variables = 5000 

One problem that occured was that the decision variable x_513 was missing. I simply assumed the cost matrix for the corrsponding element to be 0. '''



f = open(f"test1.lp",'r')
s = f.read()
f.close()

f = open(f"output.txt",'w')
l = s.split()

f.write("Processing Given Data ... \n")

# Start from 5th index to get the matrices.

n = 5000 # Decision variables
c = np.array([0 for i in range(n)])
i = 4
sign = 1

while True:
    if l[i] == '-':
        sign = -1
    elif l[i][0] == 'S': # Stop at Subject to
        break
    else:
        sign = 1

    i += 1
    num = float(l[i])
    i += 1

    ind = int(l[i][2:])
    i += 1
    c[ind] = sign*num

m = 500 # Number of constraints
i += 2

# Start with r_0

A = []
b = []

eq = 0

for k in range(m):
    p = np.array([0 for i in range(n)])

    while True:
        if l[i] == "<=":
            i += 1
            eq += 1
            b.append(float(l[i]))
            i += 1
            break

        elif l[i] == ">=":
            for j in range(n):
                p[j] *= -1
            i += 1
            b.append(-float(l[i]))
            i += 1
            break

        elif l[i] == "=":
            print('hehe')
            break

        elif l[i] == '-':
            sign = -1
        else:
            sign = 1
        i += 1

        if l[i][0] != 'x':
            num = float(l[i])
            i += 1
        else:
            num = 1
        
        ind = int(l[i][2:])
        i += 1
        p[ind] = sign*num

    A.append(p)

A = np.array(A)
b = np.array(b)

# Since eq == 500 == m, all constraints are of the form Ax <= b

# So far all data has been extracted into A,B and C

f.write("\nRank of A matrix: " + str(np.linalg.matrix_rank(A))+'\n') # Since the Rank is 500, all rows are independent.
f.write("\nData processing Done! \n")

    
############################################# EXPLORING CPLEX ####################################

from docplex.mp.model import Model

''' Tp find the first set of columns, I randomly tried a set of constraints. I took the first 1000 columns of the data set and was able to replicate a possible solution using the CPLEX api. This could now be used as the initial W.'''

def solve_primal_and_dual(A, b, c):

    primal_model = Model(name='Primal_LP')
    num_variables = A.shape[1]
    num_constraints = A.shape[0]

    x = {i: primal_model.continuous_var(name='x_{0}'.format(i)) for i in range(num_variables)} # Decision Variables

    # Objective function for primal problem
    primal_model.maximize(primal_model.sum(c[i] * x[i] for i in range(num_variables)))

    # Constraints for primal problem
    for i in range(len(b)):
        primal_model.add_constraint(primal_model.sum(A[i, j] * x[j] for j in range(num_variables)) <= b[i])

    primal_model.solve()
    primal_solution = [x[i].solution_value for i in range(num_variables)]



    # Create a new optimization model for dual problem

    dual_model = Model(name='Dual_LP')
    
    y = {i: dual_model.continuous_var(name='y_{0}'.format(i)) for i in range(num_constraints)}

    dual_model.minimize(dual_model.sum(b[i] * y[i] for i in range(num_constraints)))

    for j in range(num_variables):
        dual_model.add_constraint(dual_model.sum(A[i, j] * y[i] for i in range(num_constraints)) >= c[j])

    dual_model.solve()
    dual_solution = [y[i].solution_value for i in range(num_constraints)]

    return primal_solution, dual_solution

###############################################  SIFTING #####################################################

''' Now we formally get started with the sifting procedure.

The sifting algorithm is a simplex-based column-generation algorithm. This algorithm starts by solving a reduced problem that includes a subset of variables. At each iteration, the algorithm adds new variables and solves the problem again. The algorithm stops when the optimal solution of the current iteration is found to be optimal for the original problem.

The sifting algorithm is usually helpful for problems that have many more variables than constraints which is true in our case. From the research paper, Sifting is divided into 5 steps

1. We assume pi to be our present solution. Now, while the present solution satisfies the constraints,

2. We choose a good set of candidates which haven't been considered yet and include this in our solution structure.

3. Then we use simplex algorithm to solve this for the new solution set choosen using docplex module and CPLEX support, and repeat the entire process.

The problem formalised is: Maximize c^T x , Subject to Ax <= b 
The corresponding duel is: Minimize b^T y , Subject to A^T y >= c

'''


''' 

The initial set of columns that satisfies is the first 500 columns. Hence w = 1:500 .
Now we find the corresponding xw* and pi*

'''

def check(A,c,pi_star): # Gives me an array of the form [A^T pi*/c,col no] so that i can sort this and find the columns with neg value and add them to my w set.

    rho = []

    for row in range(n):
        temp = 0
        for j in range(len(pi_star)):
            temp += A[j][row]*pi_star[j]
        if temp<c[row]:
            rho.append([temp/c[row],row]) # Uses the Lambda Pricing Algorithm
    
    return rho


w = set([i for i in range(500)]) # Initial w is first 500 columns. It has been verified that this has a valid solution

Aw = np.array([[p[i] for i in w] for p in A]) # Taking the w columns of A and C
Cw = np.array([c[i] for i in w])

xw_star , pi_star = solve_primal_and_dual(Aw,b,Cw)

itr = 1
prev = 0

f.write("\nRunning Sifting process ...\n")
f.write("\nInitial set size : "+str(len(w))+'\n')

while len(w) != prev: # Shows that the set's size is increasing

    rho = check(A,c,pi_star)
    rho.sort()
    prev = len(w)

    t = 100 # The number of columns that we attempt to put in rho each time

    i = 0
    while t and i<len(rho):
        if rho[i][1] not in w:
            w.add(rho[i][1])
            t-=1
        i += 1

    Aw = np.array([[p[i] for i in w] for p in A]) # Taking the w columns of A and C
    Cw = np.array([c[i] for i in w])

    xw_star , pi_star = solve_primal_and_dual(Aw,b,Cw)

    f.write("\nIteration no: "+str(itr))
    f.write("\nCurrent set size: " + str(len(w))+'\n') # To make sure we don't cross 1000 which is the limit of the community CPLEX version
    
    itr += 1
    
solution = [0 for i in range(n)]
ptr = 0

for i in w:
    solution[i] = xw_star[ptr]
    ptr += 1
    
f.write("\nFinal Solution: \n")

for i in range(n):
    f.write(f'\nx_{i} : {solution[i]}')

ans = 0 # Find ovjective function value

for i in range(n):
    ans += solution[i]*c[i]

f.write("\nMax objective function Value: "+str(ans))


############################################# CHECKER ##########################################

''' This part checks if the solution we have produced is valid.'''

f.write("\nChecking the solution...\n")

flag = True
cnt = 0
error = 1e-9
for i in range(m):
    temp = 0
    for j in range(n):
        temp += A[i][j]*solution[j]
    if temp>b[i]+error:
        flag = False

'''
    I've used b[i]+error in place of b[i] in the checking part. This is to avoid minute errors in the order of 1e-9, which occur in the process of multipication
'''
        
if flag:
    f.write("Solution has been checked and is true! ")
else:
    f.write(cnt," Constraints have been violated")

f.close()
