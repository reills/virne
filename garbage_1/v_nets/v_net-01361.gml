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
  id 1361
  arrival_time 28725.537401733116
  lifetime 1162.688418381891
  num_nodes 7
  type "path"
  node [
    id 0
    label "0"
    cpu 8
    gpu 28
    rom 26
  ]
  node [
    id 1
    label "1"
    cpu 34
    gpu 3
    rom 11
  ]
  node [
    id 2
    label "2"
    cpu 22
    gpu 0
    rom 34
  ]
  node [
    id 3
    label "3"
    cpu 50
    gpu 44
    rom 38
  ]
  node [
    id 4
    label "4"
    cpu 37
    gpu 8
    rom 44
  ]
  node [
    id 5
    label "5"
    cpu 28
    gpu 17
    rom 3
  ]
  node [
    id 6
    label "6"
    cpu 12
    gpu 39
    rom 50
  ]
  edge [
    source 0
    target 1
    bw 48
  ]
  edge [
    source 1
    target 2
    bw 39
  ]
  edge [
    source 2
    target 3
    bw 14
  ]
  edge [
    source 3
    target 4
    bw 14
  ]
  edge [
    source 4
    target 5
    bw 36
  ]
  edge [
    source 5
    target 6
    bw 32
  ]
]
