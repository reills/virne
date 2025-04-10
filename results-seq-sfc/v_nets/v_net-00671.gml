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
  id 671
  arrival_time 14329.312906955456
  lifetime 550.885697580516
  num_nodes 12
  type "path"
  node [
    id 0
    label "0"
    cpu 21
    gpu 0
    rom 8
  ]
  node [
    id 1
    label "1"
    cpu 23
    gpu 49
    rom 46
  ]
  node [
    id 2
    label "2"
    cpu 36
    gpu 15
    rom 9
  ]
  node [
    id 3
    label "3"
    cpu 11
    gpu 30
    rom 50
  ]
  node [
    id 4
    label "4"
    cpu 26
    gpu 4
    rom 46
  ]
  node [
    id 5
    label "5"
    cpu 42
    gpu 29
    rom 17
  ]
  node [
    id 6
    label "6"
    cpu 16
    gpu 35
    rom 26
  ]
  node [
    id 7
    label "7"
    cpu 27
    gpu 40
    rom 28
  ]
  node [
    id 8
    label "8"
    cpu 32
    gpu 20
    rom 2
  ]
  node [
    id 9
    label "9"
    cpu 9
    gpu 40
    rom 11
  ]
  node [
    id 10
    label "10"
    cpu 12
    gpu 15
    rom 11
  ]
  node [
    id 11
    label "11"
    cpu 5
    gpu 15
    rom 44
  ]
  edge [
    source 0
    target 1
    bw 30
  ]
  edge [
    source 1
    target 2
    bw 35
  ]
  edge [
    source 2
    target 3
    bw 16
  ]
  edge [
    source 3
    target 4
    bw 20
  ]
  edge [
    source 4
    target 5
    bw 38
  ]
  edge [
    source 5
    target 6
    bw 42
  ]
  edge [
    source 6
    target 7
    bw 12
  ]
  edge [
    source 7
    target 8
    bw 14
  ]
  edge [
    source 8
    target 9
    bw 20
  ]
  edge [
    source 9
    target 10
    bw 36
  ]
  edge [
    source 10
    target 11
    bw 39
  ]
]
