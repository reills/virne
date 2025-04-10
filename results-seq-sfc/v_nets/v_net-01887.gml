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
  id 1887
  arrival_time 41563.22578591951
  lifetime 1963.1709046968706
  num_nodes 9
  type "path"
  node [
    id 0
    label "0"
    cpu 47
    gpu 4
    rom 28
  ]
  node [
    id 1
    label "1"
    cpu 25
    gpu 23
    rom 44
  ]
  node [
    id 2
    label "2"
    cpu 42
    gpu 49
    rom 2
  ]
  node [
    id 3
    label "3"
    cpu 22
    gpu 33
    rom 35
  ]
  node [
    id 4
    label "4"
    cpu 33
    gpu 8
    rom 11
  ]
  node [
    id 5
    label "5"
    cpu 27
    gpu 35
    rom 40
  ]
  node [
    id 6
    label "6"
    cpu 30
    gpu 6
    rom 26
  ]
  node [
    id 7
    label "7"
    cpu 12
    gpu 49
    rom 16
  ]
  node [
    id 8
    label "8"
    cpu 9
    gpu 43
    rom 41
  ]
  edge [
    source 0
    target 1
    bw 44
  ]
  edge [
    source 1
    target 2
    bw 46
  ]
  edge [
    source 2
    target 3
    bw 10
  ]
  edge [
    source 3
    target 4
    bw 50
  ]
  edge [
    source 4
    target 5
    bw 49
  ]
  edge [
    source 5
    target 6
    bw 5
  ]
  edge [
    source 6
    target 7
    bw 19
  ]
  edge [
    source 7
    target 8
    bw 35
  ]
]
