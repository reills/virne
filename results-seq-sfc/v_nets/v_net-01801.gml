graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 1801
  arrival_time 39985.15580143646
  lifetime 1224.972456715719
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 2
    gpu 12
    rom 4
  ]
  node [
    id 1
    label "1"
    cpu 32
    gpu 39
    rom 22
  ]
  node [
    id 2
    label "2"
    cpu 19
    gpu 43
    rom 48
  ]
  node [
    id 3
    label "3"
    cpu 42
    gpu 28
    rom 7
  ]
  node [
    id 4
    label "4"
    cpu 9
    gpu 9
    rom 33
  ]
  node [
    id 5
    label "5"
    cpu 41
    gpu 13
    rom 44
  ]
  node [
    id 6
    label "6"
    cpu 41
    gpu 18
    rom 25
  ]
  node [
    id 7
    label "7"
    cpu 32
    gpu 19
    rom 49
  ]
  node [
    id 8
    label "8"
    cpu 20
    gpu 15
    rom 15
  ]
  node [
    id 9
    label "9"
    cpu 42
    gpu 8
    rom 9
  ]
  edge [
    source 0
    target 1
    bw 10
  ]
  edge [
    source 1
    target 2
    bw 33
  ]
  edge [
    source 2
    target 3
    bw 47
  ]
  edge [
    source 3
    target 4
    bw 22
  ]
  edge [
    source 4
    target 5
    bw 38
  ]
  edge [
    source 5
    target 6
    bw 31
  ]
  edge [
    source 6
    target 7
    bw 37
  ]
  edge [
    source 7
    target 8
    bw 45
  ]
  edge [
    source 8
    target 9
    bw 21
  ]
]
