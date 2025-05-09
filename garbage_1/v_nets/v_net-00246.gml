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
  id 246
  arrival_time 4546.113401304444
  lifetime 30.94934713902207
  num_nodes 8
  type "path"
  node [
    id 0
    label "0"
    cpu 38
    gpu 22
    rom 16
  ]
  node [
    id 1
    label "1"
    cpu 50
    gpu 9
    rom 48
  ]
  node [
    id 2
    label "2"
    cpu 25
    gpu 21
    rom 24
  ]
  node [
    id 3
    label "3"
    cpu 12
    gpu 7
    rom 12
  ]
  node [
    id 4
    label "4"
    cpu 26
    gpu 27
    rom 27
  ]
  node [
    id 5
    label "5"
    cpu 30
    gpu 36
    rom 19
  ]
  node [
    id 6
    label "6"
    cpu 36
    gpu 21
    rom 15
  ]
  node [
    id 7
    label "7"
    cpu 28
    gpu 10
    rom 0
  ]
  edge [
    source 0
    target 1
    bw 29
  ]
  edge [
    source 1
    target 2
    bw 24
  ]
  edge [
    source 2
    target 3
    bw 7
  ]
  edge [
    source 3
    target 4
    bw 44
  ]
  edge [
    source 4
    target 5
    bw 34
  ]
  edge [
    source 5
    target 6
    bw 14
  ]
  edge [
    source 6
    target 7
    bw 26
  ]
]
