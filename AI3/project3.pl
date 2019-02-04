%Appending an item to the list
append([], L, L).
append([Head|Tail], Middle, [Head|Tail2]):- append(Tail , Middle, Tail2).

%recursive predicate to fetch the role hierarchy	
descendant(X, Y, _) :- rh(X, Y).
descendant(X, Y, Visited) :- 
	rh(X, Z),
	\+ member(Z,Visited) -> descendant(Z,Y,[Z|Visited]).
descendants(X, L) :- findall(A, descendant(X, A, [X]), L).
  
%Generate a single list from list of lists
flatten_list(X,[X]) :- \+ is_list(X).
flatten_list([],[]).
flatten_list([X|Xs],Zs) :- flatten_list(X,Y), flatten_list(Xs,Ys), append(Y,Ys,Zs).

%predicate to delete an item from a list
delete([], _, []).
delete([Elem|Tail], Del, Result) :-
     (   \+ Elem \= Del
     ->  delete(Tail, Del, Result)
     ;   Result = [Elem|Rest],
          delete(Tail, Del, Rest)
     ).


%recursive implementation to sort a list of numbers
quicksort([], []).
quicksort([X0|Xs], Ys) :-
	partition(X0, Xs, Ls, Gs),
	quicksort(Ls, Ys1),
	quicksort(Gs, Ys2),
	append(Ys1, [X0|Ys2], Ys).

%quick sort takes help of this predicate to get the partitioned list and sorts them.
partition(Pivot,[],[],[]).
partition(Pivot,[X|Xs],[X|Ys],Zs) :-
	X =< Pivot,
	!, % cut
	partition(Pivot,Xs,Ys,Zs).
partition(Pivot,[X|Xs],Ys,[X|Zs]) :-
	partition(Pivot,Xs,Ys,Zs). 

%predicate to check whether a element is existing in in a list.	
member(El, [H|T]) :-
  member_(T, El, H).
  member_(_, El, El).
  member_([H|T], El, _) :-
  member_(T, El, H).

%uniques removes duplicates from the list i.e. converts a list to a set.
uniques([],X,X).
uniques([C|D], F,L) :-
    \+ member(C,F) -> uniques(D, [C|F], L);
    uniques(D,F,L).

%fetch all the roles for a given user.	
authorized_roles(User, ListRoles1):-
	findRoles(User, ListRoles),
	flatten_list(ListRoles,Set),
    uniques(Set, [], ListRoles1).

%fetches the role and its hierarchy descendants for the user.	
findRole(User, FL):-
	ur(User,R),
	descendants(R, L),
	FL = [R|L].

%populates all the roles fetched into a List L.
findRoles(User, L):-
	findall(Rolelist,findRole(User,Rolelist), L).
	
%finds the permissions related to all the roles passed to this predicate as the parameter
findPermissions([],[]).
findPermissions([Role|RolesList],[R|X]) :-
    findall(Perm, rp(Role,Perm), R),
    %append([Perm],L,L),
    findPermissions(RolesList,X).

%finds all the unique permissions of a User based on the roles retrieved from the authorized_roles which handles hierarchies.
authorized_permissions(User,L) :-
    authorized_roles(User, ListRoles),
    findPermissions(ListRoles, ListPerms),
	flatten_list(ListPerms,Set),
    uniques(Set, [], L).

% returns the length of a list 	
len(0,[]).
len(L+1, [H|T]) :- len(L,T).

% question 3. Returns the minimum roles needed to create the single layered role structure.
minRoles(S):-
	users(U),
	findUsers(1,U,L),
	findPermissionsofUsers(L,[],P),
	uniques(P,[],Set),
	len(SLen, Set),S is SLen.

% returns the permissions of all the users in the system.
findPermissionsofUsers([],S,X):- X = S.
findPermissionsofUsers([H|L],S,X):-
	authorized_permissions(H,List),
	quicksort(List,Y),
	findPermissionsofUsers(L,[Y|S], X).

%below two predicates gives all the users from 1 to X.	
findUsers(L, U, Ns) :-
    L =< U,
    putUsersIntoList(L, U, Ns).
 
putUsersIntoList(N, N, L) :- !,L = [N].
putUsersIntoList(L, N, [L|Nums]) :-
    L2 is L+1,
    putUsersIntoList(L2, N, Nums).
