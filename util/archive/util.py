
def has_converged(template_path, molpro_out_path='MOLPRO/molpro.out'):
    """has molpro converged?"""
    with open(molpro_out_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'Final alpha occupancy' in line or 'Final occupancy' in line:
                final_iteration_no = int(re.search(r'\d+', lines[i - 2]).group())

    # print(final_iteration_no)
    maxit = 60  # the default
    with open(template_path, 'r') as f:
        for line in f:
            if 'maxit' in line:
                maxit = line.rstrip().split('maxit=')[1]  # take the non-default if present in the input
                break

    # print(maxit)
    if maxit == final_iteration_no:
        print(f'Final iteration no was found to be {maxit}, optimisation has not converged')
        return False
    else:
        return True