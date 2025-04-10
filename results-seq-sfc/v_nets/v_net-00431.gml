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
  id 431
  arrival_time 8406.91216402508
  lifetime 337.1114412122913
  num_nodes 9
  type "path"
  node [
    id 0
    label "0"
    cpu 38
    gpu 43
    rom 14
  ]
  node [
    id 1
    label "1"
    cpu 46
    gpu 31
    rom 42
  ]
  node [
    id 2
    label "2"
    cpu 42
    gpu 16
    rom 0
  ]
  node [
    id 3
    label "3"
    cpu 4
    gpu 18
    rom 14
  ]
  node [
    id 4
    label "4"
    cpu 25
    gpu 29
    rom 32
  ]
  node [
    id 5
    label "5"
    cpu 37
    gpu 32
    rom 35
  ]
  node [
    id 6
    label "6"
    cpu 9
    gpu 22
    rom 24
  ]
  node [
    id 7
    label "7"
    cpu 36
    gpu 7
    rom 30
  ]
  node [
    id 8
    label "8"
    cpu 18
    gpu 14
    rom 10
  ]
  edge [
    source 0
    target 1
    bw 11
  ]
  edge [
    source 1
    target 2
    bw 45
  ]
  edge [
    source 2
    target 3
    bw 28
  ]
  edge [
    source 3
    target 4
    bw 17
  ]
  edge [
    source 4
    target 5
    bw 42
  ]
  edge [
    source 5
    target 6
    bw 24
  ]
  edge [
    source 6
    target 7
    bw 36
  ]
  edge [
    source 7
    target 8
    bw 14
  ]
]
