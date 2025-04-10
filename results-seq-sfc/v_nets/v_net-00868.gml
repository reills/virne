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
  id 868
  arrival_time 18103.603177021818
  lifetime 606.2538343358692
  num_nodes 15
  type "path"
  node [
    id 0
    label "0"
    cpu 13
    gpu 26
    rom 28
  ]
  node [
    id 1
    label "1"
    cpu 14
    gpu 37
    rom 2
  ]
  node [
    id 2
    label "2"
    cpu 42
    gpu 3
    rom 48
  ]
  node [
    id 3
    label "3"
    cpu 11
    gpu 13
    rom 44
  ]
  node [
    id 4
    label "4"
    cpu 13
    gpu 2
    rom 38
  ]
  node [
    id 5
    label "5"
    cpu 37
    gpu 9
    rom 25
  ]
  node [
    id 6
    label "6"
    cpu 38
    gpu 0
    rom 30
  ]
  node [
    id 7
    label "7"
    cpu 8
    gpu 35
    rom 50
  ]
  node [
    id 8
    label "8"
    cpu 2
    gpu 37
    rom 4
  ]
  node [
    id 9
    label "9"
    cpu 25
    gpu 7
    rom 1
  ]
  node [
    id 10
    label "10"
    cpu 35
    gpu 4
    rom 47
  ]
  node [
    id 11
    label "11"
    cpu 33
    gpu 38
    rom 33
  ]
  node [
    id 12
    label "12"
    cpu 44
    gpu 23
    rom 13
  ]
  node [
    id 13
    label "13"
    cpu 2
    gpu 23
    rom 43
  ]
  node [
    id 14
    label "14"
    cpu 26
    gpu 11
    rom 24
  ]
  edge [
    source 0
    target 1
    bw 19
  ]
  edge [
    source 1
    target 2
    bw 0
  ]
  edge [
    source 2
    target 3
    bw 50
  ]
  edge [
    source 3
    target 4
    bw 6
  ]
  edge [
    source 4
    target 5
    bw 13
  ]
  edge [
    source 5
    target 6
    bw 4
  ]
  edge [
    source 6
    target 7
    bw 18
  ]
  edge [
    source 7
    target 8
    bw 16
  ]
  edge [
    source 8
    target 9
    bw 13
  ]
  edge [
    source 9
    target 10
    bw 38
  ]
  edge [
    source 10
    target 11
    bw 11
  ]
  edge [
    source 11
    target 12
    bw 44
  ]
  edge [
    source 12
    target 13
    bw 6
  ]
  edge [
    source 13
    target 14
    bw 2
  ]
]
