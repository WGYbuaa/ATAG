In the demo, the numbers 0~11 indicate the initial test function, 
the list indicates the high-level test function, 
and the numbers in the list indicate the initial function called,
where the level indicate the calling level.


Result:
1.High-level functions generated by ATAGcf:
[0, 1, 2]
[6, 7, 8]

2.High-level functions generated by ATAGdf:
[[2], [3]]
[[2, [3]], [4]]
[[8], [9]]
[[8, [9]], [10]]

3.High-level functions generated by ATAGcf+df:
[0, 1, 2]
[[0, [1], [2]], [3]]
[[0, [1], [2], [3]], [4]]
[6, 7, 8]
[[6, [7], [8]], [9]]
[[6, [7], [8], [9]], [10]]