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
  id 1506
  arrival_time 33583.315702197855
  lifetime 546.3391027500154
  num_nodes 12
  type "path"
  node [
    id 0
    label "0"
    cpu 42
    gpu 8
    rom 47
  ]
  node [
    id 1
    label "1"
    cpu 36
    gpu 25
    rom 0
  ]
  node [
    id 2
    label "2"
    cpu 37
    gpu 45
    rom 27
  ]
  node [
    id 3
    label "3"
    cpu 11
    gpu 16
    rom 33
  ]
  node [
    id 4
    label "4"
    cpu 11
    gpu 26
    rom 21
  ]
  node [
    id 5
    label "5"
    cpu 11
    gpu 29
    rom 45
  ]
  node [
    id 6
    label "6"
    cpu 33
    gpu 18
    rom 25
  ]
  node [
    id 7
    label "7"
    cpu 13
    gpu 38
    rom 15
  ]
  node [
    id 8
    label "8"
    cpu 12
    gpu 2
    rom 36
  ]
  node [
    id 9
    label "9"
    cpu 40
    gpu 16
    rom 26
  ]
  node [
    id 10
    label "10"
    cpu 19
    gpu 35
    rom 5
  ]
  node [
    id 11
    label "11"
    cpu 25
    gpu 6
    rom 18
  ]
  edge [
    source 0
    target 1
    bw 2
  ]
  edge [
    source 1
    target 2
    bw 23
  ]
  edge [
    source 2
    target 3
    bw 43
  ]
  edge [
    source 3
    target 4
    bw 28
  ]
  edge [
    source 4
    target 5
    bw 24
  ]
  edge [
    source 5
    target 6
    bw 24
  ]
  edge [
    source 6
    target 7
    bw 29
  ]
  edge [
    source 7
    target 8
    bw 29
  ]
  edge [
    source 8
    target 9
    bw 5
  ]
  edge [
    source 9
    target 10
    bw 44
  ]
  edge [
    source 10
    target 11
    bw 21
  ]
]
