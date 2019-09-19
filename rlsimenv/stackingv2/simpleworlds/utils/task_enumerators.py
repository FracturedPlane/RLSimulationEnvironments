from itertools import permutations, combinations, chain, product
import numpy as np
import pickle
import pdb

def enumerate_tasks(num_objs=4, length=3):
    tasks = _enumerate_tasks(num_objs=num_objs, length=length)

    # Get rid of duplicates, oops...
    tasks = [tuple(x) for x in tasks]
    tasks = list(set(tasks))

    return tasks

# Recursive enumerator, finds all paths in a "DFA"
def _enumerate_tasks(num_objs=4, length=3, task=[], phase=0, curr_obj=0, tasks=[]):
    '''
    Maintain a list of `tasks` to do, also maintain a `task` to which
    we append things and add a copy to `tasks`. The idea here is
    when constructing a task we can at each point make a decision
    about what to add. We simply exectue all possible decisions and
    append the result to `tasks` to get all possible tasks
    '''

    # Copy: use by value, not reference
    task = task[:]

    if length==0 or curr_obj == num_objs:
        return

    # Place object in box
    if phase==0:
        # Place any of num_obj objects in the box
        for obj in range(num_objs):
            new_task = [(2, (obj, ))]
            tasks += (task + new_task, )
            _enumerate_tasks(num_objs=num_objs,
                            length=length-1,
                            task=new_task,
                            phase=phase+1,
                            curr_obj=obj+1,
                            tasks=tasks)

        # Or don't place any objects in the box
        _enumerate_tasks(num_objs=num_objs,
                        length=length,
                        task=task,
                        phase=phase+1,
                        curr_obj=curr_obj,
                        tasks=tasks)

    # Place objects in corner
    if phase==1:
        # Either place object in a corner, 
        # skip the object, or skip placing in corner

        # Place in one of four corners
        for corner in [[0,0],[0,1],[1,0],[1,1]]:
            new_task = [(0, (curr_obj, corner[0], corner[1]))]
            tasks += (task + new_task, )
            _enumerate_tasks(num_objs=num_objs,
                            length=length-1,
                            task=task+new_task,
                            phase=phase,
                            curr_obj=curr_obj+1,
                            tasks=tasks)

        # Skip this object
        _enumerate_tasks(num_objs=num_objs,
                        length=length,
                        task=task,
                        phase=phase,
                        curr_obj=curr_obj+1,
                        tasks=tasks)

        # Or skip this task
        _enumerate_tasks(num_objs=num_objs,
                        length=length,
                        task=task,
                        phase=phase+1,
                        curr_obj=curr_obj,
                        tasks=tasks)

    # Stack objects
    if phase==2:
        # Don't pick up objects that have things on it
        dont_stack = set([x[1][1] for x in task if x[0] == 1])

        # If we can stack this on something
        if curr_obj not in dont_stack:
            # Find a destination that isn't itself
            for dest_obj in range(num_objs):
                if dest_obj != curr_obj:
                    new_task = [(1, (curr_obj, dest_obj))]
                    tasks += (task + new_task, )
                    _enumerate_tasks(num_objs=num_objs,
                                    length=length-1,
                                    task=task+new_task,
                                    phase=phase,
                                    curr_obj=curr_obj+1,
                                    tasks=tasks)

        # Or skip this object
        _enumerate_tasks(num_objs=num_objs,
                        length=length,
                        task=task,
                        phase=phase,
                        curr_obj=curr_obj+1,
                        tasks=tasks)

    return tasks

def enumerate_tasks_corners(num_corners=4, ordered=True):
    num_objs = self.object_dim
    max_length = self.max_length
    objs = list(range(num_objs))

    subsets = []
    for n in range(1, max_length + 1):
        subsets += list(combinations(objs, n))

    ordered_objs_list = [list(permutations(x)) for x in subsets]
    ordered_objs_list = list(chain.from_iterable(ordered_objs_list))

    # Order the objs
    if ordered:
        is_ordered = lambda x: len(x) == 1 or min(np.diff(x)) > 0
        ordered_objs_list = [x for x in ordered_objs_list if is_ordered(x)]

    tasks = []

    # List all corners, choose only up to `num_corners` of them
    corners = [(0,0), (0,1), (1,0), (1,1)]
    corners = corner[:num_corners]

    for ordered_objs in ordered_objs_list:
        cs = list(product(*[corners]*len(ordered_objs)))
        for c in cs:
            task = ()
            for obj, x in zip(ordered_objs, c):
                task += ((0, (obj,) + x),)
            tasks.append(task)

    return tasks

'''
tasks = enumerate_tasks()

for task in tasks:
    print(task)
    input()
'''
