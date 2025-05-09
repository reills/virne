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
  id 553
  arrival_time 10393.225167087803
  lifetime 3428.8640943403393
  num_nodes 12
  type "path"
  node [
    id 0
    label "0"
    cpu 11
    gpu 4
    rom 1
  ]
  node [
    id 1
    label "1"
    cpu 8
    gpu 44
    rom 7
  ]
  node [
    id 2
    label "2"
    cpu 35
    gpu 30
    rom 41
  ]
  node [
    id 3
    label "3"
    cpu 38
    gpu 38
    rom 44
  ]
  node [
    id 4
    label "4"
    cpu 18
    gpu 0
    rom 28
  ]
  node [
    id 5
    label "5"
    cpu 19
    gpu 37
    rom 6
  ]
  node [
    id 6
    label "6"
    cpu 29
    gpu 46
    rom 35
  ]
  node [
    id 7
    label "7"
    cpu 49
    gpu 28
    rom 40
  ]
  node [
    id 8
    label "8"
    cpu 6
    gpu 47
    rom 46
  ]
  node [
    id 9
    label "9"
    cpu 11
    gpu 40
    rom 30
  ]
  node [
    id 10
    label "10"
    cpu 24
    gpu 38
    rom 28
  ]
  node [
    id 11
    label "11"
    cpu 16
    gpu 12
    rom 38
  ]
  edge [
    source 0
    target 1
    bw 1
  ]
  edge [
    source 1
    target 2
    bw 46
  ]
  edge [
    source 2
    target 3
    bw 26
  ]
  edge [
    source 3
    target 4
    bw 36
  ]
  edge [
    source 4
    target 5
    bw 30
  ]
  edge [
    source 5
    target 6
    bw 9
  ]
  edge [
    source 6
    target 7
    bw 18
  ]
  edge [
    source 7
    target 8
    bw 45
  ]
  edge [
    source 8
    target 9
    bw 40
  ]
  edge [
    source 9
    target 10
    bw 7
  ]
  edge [
    source 10
    target 11
    bw 23
  ]
]
