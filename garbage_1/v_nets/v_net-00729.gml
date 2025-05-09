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
  id 729
  arrival_time 15339.368240841857
  lifetime 313.94872253020867
  num_nodes 6
  type "path"
  node [
    id 0
    label "0"
    cpu 9
    gpu 6
    rom 40
  ]
  node [
    id 1
    label "1"
    cpu 42
    gpu 47
    rom 0
  ]
  node [
    id 2
    label "2"
    cpu 32
    gpu 19
    rom 45
  ]
  node [
    id 3
    label "3"
    cpu 14
    gpu 11
    rom 11
  ]
  node [
    id 4
    label "4"
    cpu 48
    gpu 40
    rom 38
  ]
  node [
    id 5
    label "5"
    cpu 10
    gpu 6
    rom 37
  ]
  edge [
    source 0
    target 1
    bw 50
  ]
  edge [
    source 1
    target 2
    bw 4
  ]
  edge [
    source 2
    target 3
    bw 7
  ]
  edge [
    source 3
    target 4
    bw 43
  ]
  edge [
    source 4
    target 5
    bw 21
  ]
]
