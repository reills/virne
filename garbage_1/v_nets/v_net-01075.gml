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
  id 1075
  arrival_time 22478.409959736782
  lifetime 829.1274276955393
  num_nodes 7
  type "path"
  node [
    id 0
    label "0"
    cpu 2
    gpu 34
    rom 4
  ]
  node [
    id 1
    label "1"
    cpu 10
    gpu 47
    rom 32
  ]
  node [
    id 2
    label "2"
    cpu 6
    gpu 28
    rom 33
  ]
  node [
    id 3
    label "3"
    cpu 15
    gpu 13
    rom 28
  ]
  node [
    id 4
    label "4"
    cpu 40
    gpu 17
    rom 44
  ]
  node [
    id 5
    label "5"
    cpu 6
    gpu 37
    rom 8
  ]
  node [
    id 6
    label "6"
    cpu 0
    gpu 27
    rom 0
  ]
  edge [
    source 0
    target 1
    bw 30
  ]
  edge [
    source 1
    target 2
    bw 25
  ]
  edge [
    source 2
    target 3
    bw 20
  ]
  edge [
    source 3
    target 4
    bw 4
  ]
  edge [
    source 4
    target 5
    bw 50
  ]
  edge [
    source 5
    target 6
    bw 45
  ]
]
