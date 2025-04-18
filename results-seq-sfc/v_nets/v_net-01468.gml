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
  id 1468
  arrival_time 31933.151034957795
  lifetime 1608.3285331144007
  num_nodes 15
  type "path"
  node [
    id 0
    label "0"
    cpu 22
    gpu 19
    rom 44
  ]
  node [
    id 1
    label "1"
    cpu 10
    gpu 12
    rom 41
  ]
  node [
    id 2
    label "2"
    cpu 0
    gpu 9
    rom 4
  ]
  node [
    id 3
    label "3"
    cpu 24
    gpu 36
    rom 20
  ]
  node [
    id 4
    label "4"
    cpu 40
    gpu 34
    rom 3
  ]
  node [
    id 5
    label "5"
    cpu 44
    gpu 48
    rom 33
  ]
  node [
    id 6
    label "6"
    cpu 38
    gpu 7
    rom 50
  ]
  node [
    id 7
    label "7"
    cpu 30
    gpu 2
    rom 30
  ]
  node [
    id 8
    label "8"
    cpu 14
    gpu 37
    rom 22
  ]
  node [
    id 9
    label "9"
    cpu 16
    gpu 12
    rom 15
  ]
  node [
    id 10
    label "10"
    cpu 31
    gpu 5
    rom 42
  ]
  node [
    id 11
    label "11"
    cpu 40
    gpu 44
    rom 50
  ]
  node [
    id 12
    label "12"
    cpu 26
    gpu 19
    rom 29
  ]
  node [
    id 13
    label "13"
    cpu 12
    gpu 37
    rom 18
  ]
  node [
    id 14
    label "14"
    cpu 34
    gpu 1
    rom 4
  ]
  edge [
    source 0
    target 1
    bw 21
  ]
  edge [
    source 1
    target 2
    bw 16
  ]
  edge [
    source 2
    target 3
    bw 31
  ]
  edge [
    source 3
    target 4
    bw 4
  ]
  edge [
    source 4
    target 5
    bw 45
  ]
  edge [
    source 5
    target 6
    bw 10
  ]
  edge [
    source 6
    target 7
    bw 8
  ]
  edge [
    source 7
    target 8
    bw 5
  ]
  edge [
    source 8
    target 9
    bw 36
  ]
  edge [
    source 9
    target 10
    bw 44
  ]
  edge [
    source 10
    target 11
    bw 11
  ]
  edge [
    source 11
    target 12
    bw 12
  ]
  edge [
    source 12
    target 13
    bw 42
  ]
  edge [
    source 13
    target 14
    bw 43
  ]
]
