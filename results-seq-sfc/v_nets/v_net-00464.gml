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
  id 464
  arrival_time 8760.195708020616
  lifetime 500.6269920277514
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 25
    gpu 11
    rom 49
  ]
  node [
    id 1
    label "1"
    cpu 26
    gpu 25
    rom 31
  ]
  node [
    id 2
    label "2"
    cpu 1
    gpu 47
    rom 25
  ]
  node [
    id 3
    label "3"
    cpu 33
    gpu 42
    rom 11
  ]
  node [
    id 4
    label "4"
    cpu 44
    gpu 1
    rom 14
  ]
  node [
    id 5
    label "5"
    cpu 15
    gpu 36
    rom 16
  ]
  node [
    id 6
    label "6"
    cpu 3
    gpu 40
    rom 7
  ]
  node [
    id 7
    label "7"
    cpu 22
    gpu 45
    rom 13
  ]
  node [
    id 8
    label "8"
    cpu 34
    gpu 35
    rom 13
  ]
  node [
    id 9
    label "9"
    cpu 5
    gpu 20
    rom 13
  ]
  node [
    id 10
    label "10"
    cpu 7
    gpu 11
    rom 35
  ]
  edge [
    source 0
    target 1
    bw 18
  ]
  edge [
    source 1
    target 2
    bw 29
  ]
  edge [
    source 2
    target 3
    bw 15
  ]
  edge [
    source 3
    target 4
    bw 39
  ]
  edge [
    source 4
    target 5
    bw 43
  ]
  edge [
    source 5
    target 6
    bw 0
  ]
  edge [
    source 6
    target 7
    bw 14
  ]
  edge [
    source 7
    target 8
    bw 16
  ]
  edge [
    source 8
    target 9
    bw 45
  ]
  edge [
    source 9
    target 10
    bw 19
  ]
]
